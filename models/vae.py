import torch
import torch.nn as nn

from configs.model import VAEConfig
from models.activations import get_activation
from models.output import VAEOutput

class VAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config

        input_dim = int(torch.prod(torch.tensor(config.input_shape)))

        #Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, config.hidden_dim),
            get_activation(config.activation),
            nn.Linear(config.hidden_dim, config.latent_dim * 2)
        )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            get_activation(config.activation),
            nn.Linear(config.hidden_dim, input_dim),
            nn.Unflatten(1, config.input_shape)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        "Encoding the inputs into latent representations."
        encoder_output = self.encoder(x)

        mu, log_var = torch.chunk(
            encoder_output,
            chunks=2,
            dim=-1
        )

        return mu, log_var
    
    def decode(self, z:torch.Tensor) -> torch.Tensor:
        "Decode latent vectors into reconstruction logits."
        return self.decoder(z)
    
    def _logvar_to_std(self, log_var: torch.Tensor) -> torch.Tensor:
        "Convert log variance to standard deviation"
        return torch.exp(0.5 * log_var)
    
    def reparameterization(self, mu:torch.Tensor, std:torch.Tensor) -> torch.Tensor:
        "Sample latent vectors using the reparameterization trick."
        epsilon = torch.randn_like(std)
        return mu+std*epsilon
    
    def forward(self, x:torch.Tensor, reconstruct: bool=False) -> VAEOutput:
        #Encoder input
        mu, log_var = self.encode(x)

        #Compute Standard deviation 
        std = self._logvar_to_std(log_var)

        #Sample latent vector
        z = self.reparameterization(mu, std)

        #decode
        x_logits = self.decode(z)

        return VAEOutput(
            x_logits=x_logits,
            z=z,
            mu=mu,
            std=std,
            x_recon=torch.sigmoid(x_logits) if reconstruct else None,
        )