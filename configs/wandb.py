from dataclasses import dataclass

@dataclass
class WandBConfig:
    "Weights and Biases Configuration."
    project: str = "beta_vae_mnist"
    entity: str = None | None
    run_name: str = None | None
    mode: str = "online"