import torch

from configs.dataset import DatasetConfig
from configs.model import VAEConfig

from data.mnist import MNISTDataModule
from models.vae import VAE

from configs.training import TrainingConfig

from configs.wandb import WandBConfig
from utils.logger import WandBLogger
from dataclasses import asdict

from evaluation.evaluator import Evaluator
from training.checkpoints import CheckpointManager
from configs.evaluation_config import EvaluationConfig


def main():
    # Configs
    data_config = DatasetConfig()
    model_config = VAEConfig()
    training_config = TrainingConfig()
    wandb_config = WandBConfig()
    evaluation_config = EvaluationConfig()

    # Device
    device = torch.device(
        training_config.device
        if torch.cuda.is_available()
        else "cpu"
    )

    # Data
    datamodule = MNISTDataModule(data_config)

    test_loader = datamodule.test_dataloader()

    # Model
    model = VAE(model_config).to(device)

    # Load checkpoint
    checkpoint_manager = CheckpointManager(training_config, model_config)

    metadata = checkpoint_manager.load(model)

    print(metadata)

    experiment_config = {
        **asdict(data_config),
        **asdict(model_config),
        **asdict(training_config),
    }

    # Logger
    logger = WandBLogger(
        config=wandb_config,
        experiment_config=experiment_config,
    )

    # Evaluator
    evaluator = Evaluator(logger, evaluation_config)

    evaluator.evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        epoch=metadata["epoch"],
    )

    logger.finish()


if __name__ == "__main__":
    main()