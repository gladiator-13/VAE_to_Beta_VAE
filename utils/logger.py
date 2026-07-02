import wandb
from configs.wandb import WandBConfig


class WandBLogger:
    def __init__(self, config: WandBConfig, experiment_config: dict) -> None:

        self.config = config

        wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.run_name,
            mode=config.mode,
            config=experiment_config,
        )

    def log_metrics(self, metrics: dict, step: int) -> None:
        wandb.log(metrics, step=step)

    def watch_model(self, model) -> None:
        """This logs:
            gradients
            parameter histograms
            parameter updates"""
        if self.config.watch_model:
            wandb.watch(model, log="all")   

    def finish(self):
        wandb.finish()