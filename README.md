# Picaboo

A minimal GPT-style language model and Byte Pair Encoding (BPE) tokenizer implemented in PyTorch, with a simple training loop, dataset utilities, and MLflow experiment tracking.

## Features
- **Tokenizer (`tokenizer.py`)**: Simple BPE tokenizer with train, encode, and decode. Loads/saves `encoder.json` and `merges.bpe` (with special tokens and regex splitter persisted).
- **Model (`model.py`)**: Decoder-only Transformer (GPT-like) with masked multi-head self-attention, GELU MLP, layer norm, dropout, and weight tying. Includes a `generate` method.
- **Training (`trainer.py`)**: Training harness with gradient accumulation, cosine LR schedule with warmup, gradient clipping, validation, perplexity calculation, sample generation, checkpoints, and MLflow logging.
- **Utils (`utils.py`)**: Dataset class for language modeling with next-token prediction and AdamW optimizer configuration.
- **Datasets (`datasets/`)**: Sample `.txt` and `.epub` files for experimentation.

## Repository Structure
```
picaboo/
  datasets/              # Sample text corpora
  models/                # Saved tokenizer files: encoder.json, merges.bpe
  model.py               # Transformer language model (GPT-like)
  tokenizer.py           # BPE tokenizer implementation
  trainer.py             # Training loop and MLflow integration
  utils.py               # Dataset and optimizer helpers
  pretraining.ipynb      # Notebook for pretraining experiments
  tokenization.ipynb     # Notebook for tokenizer experiments
```

## Requirements
- Python 3.10+
- PyTorch (CUDA optional)
- regex
- mlflow

Install with:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch regex mlflow
```

If you need CUDA builds of PyTorch, follow the official instructions: `https://pytorch.org/get-started/locally/`.

## Tokenizer
The tokenizer can be trained from raw text and then used to encode/decode tokens. The repo includes a prebuilt tokenizer under `models/`.

### Load existing tokenizer
```python
from tokenizer import Tokenizer

# Loads from models/ (expects encoder.json and merges.bpe)
tokenizer = Tokenizer.load("models")
print(tokenizer.encode("stand tall!<|endoftext|>"))
print(tokenizer.decode(tokenizer.encode("stand tall!<|endoftext|>")))
```

### Train a tokenizer from scratch
```python
from tokenizer import Tokenizer

with open("datasets/combined.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialize with special tokens if desired
special_tokens = {"<|endoftext|>": 100257}
tokenizer = Tokenizer(special_tokens=special_tokens)
merges, token_counts = tokenizer.train(text=text, num_merges=30000)

# Persist files to models/
tokenizer.save("models")
```

## Model
`PicabooLM` is a decoder-only Transformer. Key hyperparameters are in `PicabooLMParams`:
- **context_length**: max sequence length
- **vocab_size**: tokenizer vocabulary size
- **num_blocks**: number of decoder blocks
- **num_heads**: attention heads
- **d_model**: model width (must be divisible by heads)
- **dropout_rate**, **bias**, **device**

Minimal forward pass:
```python
import torch
from model import PicabooLM, PicabooLMParams

params = PicabooLMParams(vocab_size=50257, context_length=512)
model = PicabooLM(params)

# Fake batch of token ids: (batch, time)
X = torch.randint(0, params.vocab_size, (2, 128))
logits = model(X)  # (2, 128, vocab_size)
```

### Text generation
```python
import torch
from model import PicabooLM, PicabooLMParams
from tokenizer import Tokenizer

tokenizer = Tokenizer.load("models")
params = PicabooLMParams(vocab_size=len(tokenizer.encoder), context_length=512)
model = PicabooLM(params)

prompt = "he is"
ctx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
out = model.generate(ctx, max_new_tokens=100, temperature=0.8, top_k=50)
print(tokenizer.decode(out[0].tolist()))
```

## Training
Use the `Trainer` to run pretraining with MLflow logging.

```python
import torch
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from model import PicabooLM, PicabooLMParams
from trainer import Trainer, TrainerParams
from utils import PicabooLMPretainingDataset, configure_adamw_optimizer

# Prepare tokenizer and data
tokenizer = Tokenizer.load("models")
with open("datasets/combined.txt", "r", encoding="utf-8") as f:
    text = f.read()
ids = tokenizer.encode(text)

context_len = 512
train_dataset = PicabooLMPretainingDataset(ids, context_size=context_len)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

# Simple val loader using the same dataset for demo
val_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=True)

# Model and optimizer
params = PicabooLMParams(vocab_size=len(tokenizer.encoder), context_length=context_len)
model = PicabooLM(params)
optimizer = configure_adamw_optimizer(
    model=model, weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), eps=1e-8, device_type=params.device
)

trainer_params = TrainerParams(
    model=model,
    optimizer=optimizer,
    loss_fn=torch.nn.CrossEntropyLoss(),
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    device=params.device,
    epochs=1,
    batch_size=8,
    save_every=1,
    checkpoints_path="checkpoints",
    gradient_accumulation_steps=4,
    total_steps=200,
    max_steps=200,
    max_learning_rate=6e-4,
    min_learning_rate=6e-5,
    warmup_steps=50,
    tokenizer=tokenizer,
)

trainer = Trainer(trainer_params)
trainer.train()
```

Notes:
- The trainer compiles the model with `torch.compile` (PyTorch 2+ recommended).
- Uses cosine decay LR with warmup; logs step/epoch metrics to MLflow.
- Saves checkpoints to `checkpoints/` and logs sample generations at each accumulation step.

## MLflow Tracking
`trainer.py` sets the tracking URI to `http://127.0.0.1:8000` and uses experiment name "Picaboo v0 Pretraining".

- To run a local MLflow tracking server:
```bash
mlflow server --host 127.0.0.1 --port 8000 --backend-store-uri ./mlruns --default-artifact-root ./mlartifacts
```
- Then open `http://127.0.0.1:8000` in your browser.
- The trainer logs metrics like `microstep_train_loss`, `step_train_loss`, `step_val_loss`, `step_perplexity_score`, plus text artifacts with sample completions and model checkpoints.

## Datasets
Place your raw text in `datasets/combined.txt` (or point to your own file). The repo includes sample texts for experimentation.

- For EPUBs, you’ll need to extract text externally or via a custom script; only `.txt` files are used in the examples here.

## Tips
- Ensure `vocab_size` in `PicabooLMParams` matches `len(tokenizer.encoder)`.
- Context length `T` must be ≤ `context_length`.
- If using CUDA, set the appropriate PyTorch build and device.

## License
This project is licensed under the terms of the LICENSE file included in this repository.
