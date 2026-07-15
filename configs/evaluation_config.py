from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    log_reconstructions: bool = True
    log_random_samples: bool = True
    log_latent_space: bool = True
    num_reconstruction_images: int = 8
    num_random_samples: int = 16
    traversal_steps: int = 11
    traversal_min: float = -3.0
    traversal_max: float = 3.0