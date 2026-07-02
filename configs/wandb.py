from dataclasses import dataclass


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    project: str = "vae-to-beta-vae"
    entity: str | None = None
    mode: str = "online"
    log_model: bool = True
    watch_model: bool = True
    run_name: str | None = None