import torch
from configs.training import TrainingConfig
from training.losses import vae_loss


class Trainer:
    def __init__(self, model, train_loader, test_loader, config: TrainingConfig) -> None:
        self.model=model
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.config=config

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
