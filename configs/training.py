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