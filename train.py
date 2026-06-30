from configs.dataset import DatasetConfig
from configs.model import VAEConfig

from data.mnist import MNISTDataModule
from models.vae import VAE

from training.losses import vae_loss

# Create configs
dataset_config = DatasetConfig()
model_config = VAEConfig()

# Create data module
data_module = MNISTDataModule(dataset_config)

# Get train loader
train_loader = data_module.train_dataloader()

# Create model
model = VAE(model_config)

# Get one batch
images, labels = next(iter(train_loader))

# Forward pass
output = model(images)

# Verify shapes
# print(images.shape)
# print(output.mu.shape)
# print(output.std.shape)
# print(output.z.shape)
# print(output.x_logits.shape)

total, recon, kl = vae_loss(
    output.x_logits,
    images,
    output.mu,
    output.log_var,
)

print(total)
print(recon)
print(kl)