import torch
import torch.nn.functional as F

def reconstruction_loss(x_logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    "Binary Cross Entropy loss."
    loss = F.binary_cross_entropy_with_logits(
        x_logits,
        x,
        reduction="sum"
    )

    return loss / x.size(0) # reconstruction loss per image

def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp(),
        dim=1
    )

    return kl.mean()

def vae_loss(x_logits: torch.Tensor, x:torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0,):
    "Compute total VAE Loss."
    recon = reconstruction_loss(
        x_logits, x
    )

    kl = kl_divergence(
        mu, log_var
    )

    total = recon + beta*kl

    return total, recon, kl