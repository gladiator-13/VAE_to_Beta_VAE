from dataclasses import dataclass

@dataclass
class VAEConfig:
    "Configurations for VAE"
    input_shape: tuple[int, int, int] = (1, 28, 28)
    hidden_dim: int = 400
    latent_dim: int = 2
    activation: str = "tanh"
    use_softplus_std: bool = False
    n_samples: int = 1