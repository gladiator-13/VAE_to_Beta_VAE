from configs.model import VAEConfig
from models.vae import VAE

config = VAEConfig()

model = VAE(config)

print(model)
