from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from configs.dataset import DatasetConfig

class MNISTDataModule:
    "Data Module for loading MNIST."

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.transform = self._create_transform()
        self.train_dataset, self.test_dataset =  self._prepare_dataset()

    def _create_transform(self):
        "Create image transformation."

        if self.config.normalize:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        return transforms.ToTensor()
        
    
    def _prepare_dataset(self):
        "Download dataset, train and test."

        train_dataset = datasets.MNIST(
            root=self.config.root,
            train=True,
            transform=self.transform,
            download=self.config.download
        )

        test_dataset = datasets.MNIST(
            root=self.config.root,
            train=False,
            transform=self.transform,
            download=self.config.download
        )

        return train_dataset, test_dataset
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

