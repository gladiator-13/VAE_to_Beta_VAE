from dataclasses import dataclass

@dataclass
class DatasetConfig:
    root: str = "./data"
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0 # Number of subprocesses for loading data
    pin_memory: bool = False # Faster CPU → GPU transfer (useful with CUDA)
    download: bool = True