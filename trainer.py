from dataclasses import dataclass, asdict
import torch
from typing import Any
import time
import math
import dataclasses
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")

@dataclass
class TrainerParams:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader
    device: str
    epochs: int
    batch_size: int
    save_every: int
    checkpoints_path: str
    gradient_accumulation_steps: int = 1
    total_steps: int = 200
    max_steps: int = 200
    max_learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 50
    tokenizer: Any = None
    
def calculate_perplexity(model, data_loader, device):
    """
    Calculate perplexity of a model on a dataset.

    Parameters:
    - model: The trained language model (e.g., an RNN, Transformer).
    - data_loader: DataLoader containing the test dataset (inputs and targets).
    - device: Device to run the computations (e.g., "cuda" or "cpu").

    Returns:
    - Perplexity (float): The perplexity of the model on the dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in data_loader:
            # Move data to the appropriate device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)  # Assumes outputs are raw logits
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)  # Log probabilities

            # Gather the log probabilities of the target tokens
            targets = targets.view(-1)  # Flatten targets
            log_probs = log_probs.view(-1, log_probs.size(-1))  # Flatten predictions
            loss = -log_probs[torch.arange(log_probs.size(0)), targets].sum()  # NLL loss

            total_loss += loss.item()
            total_tokens += targets.size(0)

    # Compute cross-entropy loss
    cross_entropy = total_loss / total_tokens

    # Calculate perplexity
    perplexity = math.exp(cross_entropy)
    return perplexity


class Trainer:
    def __init__(self, params: TrainerParams) -> None:
        # prepare the model by moving it to the device and setting it to training mode then compiling
        self.model = params.model
        self.model = self.model.to(params.device)
        self.model.train()
        self.model = torch.compile(self.model)
        self.params = params
        self.optimizer = params.optimizer
        self.loss_fn = params.loss_fn
        self.train_data = params.train_dataloader
        self.val_data = params.val_dataloader
        self.device = params.device
        self.save_every = params.save_every
        self.tokenizer = params.tokenizer
        
        # set the float32 matmul precision to high for better performance on GPUs
        torch.set_float32_matmul_precision("high")
        
        # steps counter
        self.step = 0
        self.train_loss_values = []
        self.val_loss_values = []
        self.perplexity_values = []
    
    def get_lr(self):
        if self.step < self.params.warmup_steps:
            return self.params.max_learning_rate * self.step / self.params.warmup_steps
        if self.step > self.params.max_steps:
            return self.params.min_learning_rate
        decay_ratio = (self.step - self.params.warmup_steps) / (self.params.max_steps - self.params.warmup_steps)
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(decay_ratio)))
        return (self.params.min_learning_rate + coeff * (self.params.max_learning_rate - self.params.min_learning_rate)).item()
    
    def _run_validation(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (X, Y) in enumerate(self.val_data):
                X = X.to(self.device)
                Y = Y.to(self.device)
                output = self.model(X)
                loss = self.loss_fn(output.view(-1, output.size(-1)), Y.view(-1))
                val_loss += loss.item()
        val_loss /= len(self.val_data)
        return val_loss
    
    def sample_completion(self):
        prompt_tokens = torch.tensor([self.tokenizer.encode(text="he is")]) # 1 B by 1 T
        output_tokens = self.model.generate(prompt_tokens, max_new_tokens=200)
        return self.tokenizer.decode(output_tokens.tolist()[0])

    def _run_epoch(self, epoch):
        # assert len(self.train_data) % self.params.gradient_accumulation_steps == 0, "Dataset size must be divisible by gradient accumulation steps"
        epoch_loss = 0
        accumulated_loss = 0  # To keep track of accumulated loss
        for i, (X, Y) in enumerate(self.train_data):
            t0 = time.time()
            self.model.train()
            X = X.to(self.device)
            Y = Y.to(self.device)
            # Forward pass with autocast
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                output = self.model(X)
                loss = self.loss_fn(output.view(-1, output.size(-1)), Y.view(-1))
                mlflow.log_metric("microstep_train_loss", loss.item(), step=(i+1))
            # Normalize the loss for gradient accumulation
            epoch_loss += loss.item()
            loss = loss / self.params.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            # Gradient accumulation condition
            if (i+1) % self.params.gradient_accumulation_steps == 0:
                self.step += 1
                # Clip gradients
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Adjust learning rate
                lr = self.get_lr()
                mlflow.log_metric("step_lr", lr, step=self.step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                # Update optimizer
                self.optimizer.step()
                self.train_loss_values.append(accumulated_loss)
                mlflow.log_metric("step_train_loss", accumulated_loss, step=self.step)
                # run validation
                val_loss = self._run_validation()
                self.val_loss_values.append(val_loss)
                mlflow.log_metric("step_val_loss", val_loss, step=self.step)
                perplexity = calculate_perplexity(self.model, self.val_data, self.device)
                self.perplexity_values.append(perplexity)
                mlflow.log_metric("step_perplexity_score", perplexity, step=self.step)
                sample_completion = self.sample_completion()
                mlflow.log_text(sample_completion, f"sample_completion_{self.step}.txt")
                # torch.cuda.synchronize()  # Wait for computation to finish
                dt = (time.time() - t0)
                tps = (self.params.batch_size * self.train_data.dataset.T) / dt
                print(f"Step {self.step}/{self.params.total_steps} | loss: {accumulated_loss:.2f} | norm: {norm:.4e} | dt: {dt * 1000:.2f} ms | lr: {lr:.2e} | tokens/s: {tps:.2f}")
                self.optimizer.zero_grad()
                accumulated_loss = 0
                
        # Compute average epoch loss
        epoch_loss /= len(self.train_data)
        return epoch_loss

    def _save_checkpoint(self, epoch, epoch_loss):
        mod_ckp = self.model.state_dict()
        mod_ckp = {k.replace("_orig_mod.", ""): v for k, v in mod_ckp.items()}
        checkpoint = {
            'model': mod_ckp,
            'optimizer': self.optimizer.state_dict(),
            'model_args': dataclasses.asdict(self.model.params),
            'iter_num': epoch,
            'best_val_loss': epoch_loss,
            'config': dataclasses.asdict(self.params),
        }
        epoch_checkpoint_path = f"{self.params.checkpoints_path}/checkpoint_{epoch}.pth"
        torch.save(checkpoint, epoch_checkpoint_path)
        print(f"Epoch {epoch+1} | Training checkpoint saved at {epoch_checkpoint_path}")
        # Log model to mlflow
        sample_input = torch.randint(0,2047,size=[1,1024])
        with torch.no_grad():
            output = self.model(sample_input)
            sample_output = output.numpy()
        signature = infer_signature(sample_input, sample_output)
        model_info = mlflow.pytorch.log_model(self.model, name=f"pretraining_checkpoint_{epoch}", signature=signature)
        mlflow.set_logged_model_tags(model_info.model_id, {"Pretraining Model": "GPT-3 Type Pretrained model"})

    def train(self):
        mlflow.set_experiment("Picaboo v0 Pretraining")
        with mlflow.start_run():
            mlflow.log_params(asdict(self.params))
            self.model.train()
            for epoch in range(self.params.epochs):
                epoch_train_loss = self._run_epoch(epoch)
                mlflow.log_metric("epoch_train_loss", epoch_train_loss, step=epoch+1)
                epoch_val_loss = self._run_validation()
                mlflow.log_metric("epoch_val_loss", epoch_val_loss, step=(epoch+1))
                if epoch % self.save_every == 0:
                    self._save_checkpoint(epoch, epoch_train_loss)
                print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.3f} | Val Loss: {epoch_val_loss:.3f}")
            return self.model