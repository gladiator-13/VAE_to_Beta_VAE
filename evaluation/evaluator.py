from visualization.reconstruction import plot_reconstructions
from visualization.sampling import plot_random_samples
from visualization.latent import plot_latent_space
from configs.evaluation_config import  EvaluationConfig
import torch

class Evaluator:
    "Runs evaluation and visualization routines."
    def __init__(self, logger, config:EvaluationConfig):
        self.logger=logger
        self.config=config

    def evaluate(self, model, dataloader, device, epoch: int):
        "Run Evaluation visualization"

        model.eval()

        with torch.no_grad():
            if self.config.log_reconstructions:
                recon_fig = plot_reconstructions(
                    model=model,
                    dataloader=dataloader,
                    device=device
                )

                self.logger.log_figure(
                    figure=recon_fig,
                    name="reconstructions",
                    step=epoch
                )

            if self.config.log_random_samples:
                sampling_fig = plot_random_samples(
                    model=model,
                    device=device,
                    num_samples=self.config.num_random_samples
                )

                self.logger.log_figure(
                    figure=sampling_fig,
                    name="random_samples",
                    step=epoch
                )

            if model.config.latent_dim == 2:
                latent_fig = plot_latent_space(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                )

                self.logger.log_figure(
                    figure=latent_fig,
                    name="latent_space",
                    step=epoch,
                )