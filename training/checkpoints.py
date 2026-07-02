from pathlib import Path
import torch
from configs.training import TrainingConfig


class CheckpointManager:
    """Handles saving and loading model checkpoints."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> Path:
        """Full path to the checkpoint file."""
        return self.checkpoint_dir / self.config.checkpoint_name

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
    ) -> dict:
        """Load a training checkpoint."""

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
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