import inspect
import torch
from torch.utils.data import Dataset
from typing import List

class PicabooLMPretainingDataset(Dataset):

    def __init__(self, dataset: List[int], context_size):
        self.T = context_size
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)//(self.T+1)

    def __getitem__(self, idx):
        buf = self.dataset[idx:idx+self.T+1]
        X = torch.tensor(buf[:-1])
        Y = torch.tensor(buf[1:])
        return X, Y
    
def configure_adamw_optimizer(
        model: torch.nn.Module, 
        weight_decay: float, 
        learning_rate: float,
        betas: tuple,
        eps: float, 
        device_type: str
    ):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)
    return optimizer