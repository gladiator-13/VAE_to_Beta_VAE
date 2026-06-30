from dataclasses import dataclass
import torch

@dataclass
class VAEOutput:
    x_logits: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    x_recon: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    loss_recon: torch.Tensor | None = None
    loss_kl: torch.Tensor | None = None