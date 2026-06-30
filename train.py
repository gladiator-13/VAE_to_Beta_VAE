from configs.model import VAEConfig
from models.vae import VAE
import torch

config = VAEConfig()

model = VAE(config)

x = torch.randn(8, 1, 28, 28)

mu, logvar = model.encode(x)
print(mu.shape)
print(logvar.shape)

z = torch.randn([8, 2])
x_logits = model.decode(z)

print(x_logits.shape)