from pathlib import Path
import torch
from configs.training import TrainingConfig
from configs.model import VAEConfig

class CheckpointManager:
    """Handles saving and loading model checkpoints."""

    def __init__(self, training_config: TrainingConfig, model_config: VAEConfig):
        self.training_config = training_config
        self.model_config = model_config

        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        """Full path to the checkpoint file."""
        filename = (
            f"best_model_latent{self.model_config.latent_dim}.pt")#_beta_{self.training_config.beta}.pt")
            # f"_beta{self.training_config.beta}.pt"
        # )
        return self.checkpoint_dir / filename
        

    def save(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save a training checkpoint."""

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(checkpoint, self.checkpoint_path)

        print(f"✓ Saved checkpoint: {self.checkpoint_path}")

    def load(
        self,
        model,
        optimizer=None,
        device="cpu"
    ) -> dict:
        """Load a training checkpoint."""

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        print(f"✓ Loaded checkpoint: {self.checkpoint_path}")

        return {
            "epoch": checkpoint["epoch"],
            "val_loss": checkpoint["val_loss"],
        }