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
        self.decode = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            get_activation(config.activation),
            nn.Linear(config.hidden_dim, input_dim),
            nn.Unflatten(1, config.input_shape)
        )
