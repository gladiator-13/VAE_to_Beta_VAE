from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 40
    learning_rate: int = 1e-2
    weight_decay: int = 0.0
    optimizer: str = "adam"
    beta: int = 16.0 # 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0
    device: str = "cuda"
    seed: int = 42
    image_log_frequency: int = 10
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_name: str = "best_model_latent10.pt"
    save_best_only: bool = True