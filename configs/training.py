from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: int = 1e-2
    weight_decay: int = 0.0
    optimizer: str = "adam"
    beta: int = 1.0
    device: str = "cuda"
    seed: int = 42
    image_log_frequency: int = 5
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_name: str = "best_model.pt"
    save_best_only: bool = True