import torch
from configs.training import TrainingConfig
from training.losses import vae_loss
from visualization.reconstruction import plot_reconstructions
import matplotlib.pyplot as plt
from training.checkpoints import CheckpointManager
from configs.model import VAEConfig

class Trainer:
    def __init__(self, model, train_loader, test_loader, training_config: TrainingConfig, logger, evaluator, model_config: VAEConfig) -> None:
        self.model=model
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.training_config=training_config
        self.model_config=model_config
        self.logger=logger

        self.device=torch.device(training_config.device)

        self.device = torch.device(
            training_config.device if torch.cuda.is_available() else "cpu"
        )

        self.optimizer=torch.optim.Adam(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        self.best_val_loss = float("inf")
        self.checkpoint_manager = CheckpointManager(self.training_config, self.model_config)
        self.evaluator=evaluator

    def train_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for images, _ in self.train_loader:
            images = images.to(self.device)
            
            self.optimizer.zero_grad()

            output=self.model(images)

            loss, recon, kl = vae_loss(
                output.x_logits,
                images,
                output.mu,
                output.log_var,
                beta=self.training_config.beta
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        num_batches = len(self.train_loader)

        return {
            "loss":total_loss / num_batches,
            "reconstruction": total_recon / num_batches,
            "kl": total_kl / num_batches
        }
    
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.to(self.device)
                output = self.model(images)

                loss, recon, kl = vae_loss(
                    output.x_logits,
                    images,
                    output.mu,
                    output.log_var,
                    beta=self.training_config.beta
                )

                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

        num_batches = len(self.test_loader)

        return {
            "loss":total_loss / num_batches,
            "reconstruction": total_recon / num_batches,
            "kl": total_kl / num_batches
        }
    
    def train(self):
        for epoch in range(self.training_config.epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]

                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_loss=self.best_val_loss,
                )

            self.logger.log_metrics(
                {
                    "train/loss": train_metrics["loss"],
                    "train/reconstruction": train_metrics["reconstruction"],
                    "train/kl": train_metrics["kl"],

                    "val/loss": val_metrics["loss"],
                    "val/reconstruction": val_metrics["reconstruction"],
                    "val/kl": val_metrics["kl"],
                },
                step=epoch,
            )

            if (epoch + 1) % self.training_config.image_log_frequency == 0:
                self.evaluator.evaluate(
                    model=self.model,
                    dataloader=self.test_loader,
                    device=self.device,
                    epoch=epoch,
                )

            print(
                f"Epoch {epoch + 1}/{self.training_config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.3f}"
            )    
