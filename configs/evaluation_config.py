from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    log_reconstructions: bool = True
    log_random_samples: bool = True
    log_latent_space: bool = True
    num_reconstruction_images: int = 8
    num_random_samples: int = 16