from configs.dataset import DatasetConfig
from configs.model import VAEConfig

from data.mnist import MNISTDataModule
from models.vae import VAE

from training.losses import reconstruction_loss
from training.trainer import Trainer

from configs.training import TrainingConfig

# Create configs
dataset_config = DatasetConfig()
model_config = VAEConfig()

# Create data module
data_module = MNISTDataModule(dataset_config)

# Get train_loader and test_loader
train_loader = data_module.train_dataloader()
test_loader = data_module.test_dataloader()

# Create model
model = VAE(model_config)

# Get one batch
images, labels = next(iter(train_loader))

# Forward pass
output = model(images)

training_config = TrainingConfig()

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    config=training_config
)

metrics = trainer.train_epoch()

print(metrics)