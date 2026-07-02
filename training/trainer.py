import torch
from configs.training import TrainingConfig
from training.losses import vae_loss


class Trainer:
    def __init__(self, model, train_loader, test_loader, config: TrainingConfig, logger) -> None:
        self.model=model
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.config=config
        self.logger=logger

        self.device=torch.device(config.device)

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.optimizer=torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

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
                beta=self.config.beta
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
                    beta=self.config.beta
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
        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

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

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.3f}"
            )    
