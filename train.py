import torch

from configs.dataset import DatasetConfig
from configs.model import VAEConfig

from data.mnist import MNISTDataModule
from models.vae import VAE

from training.losses import reconstruction_loss
from training.trainer import Trainer

from configs.training import TrainingConfig

from configs.wandb import WandBConfig
from utils.logger import WandBLogger
from dataclasses import asdict

def main():
    # Create configs
    dataset_config = DatasetConfig()
    model_config = VAEConfig()
    training_config = TrainingConfig()
    wandb_config = WandBConfig()

    experiment_config = {
        **asdict(dataset_config),
        **asdict(model_config),
        **asdict(training_config),
    }

    # Create data module
    data_module = MNISTDataModule(dataset_config)

    # Get train_loader and test_loader
    train_loader = data_module.train_dataloader()
    test_loader = data_module.test_dataloader()

    #Create the logger
    logger = WandBLogger(
        config=wandb_config,
        experiment_config=experiment_config,
    )

    # Create model
    model = VAE(model_config)

    logger.watch_model(model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=training_config,
        logger=logger
    )

    trainer.train()

    logger.finish()

if __name__ == "__main__":
    main()