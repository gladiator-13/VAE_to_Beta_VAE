from configs.dataset import DatasetConfig
from data.mnist import MNISTDataModule

config = DatasetConfig()

data_module = MNISTDataModule(config)

train_loader = data_module.train_dataloader()
test_loader = data_module.test_dataloader()

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
print(images.dtype)
print(images.min(), images.max())