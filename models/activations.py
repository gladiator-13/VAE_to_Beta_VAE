import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    "Return the activation function by name."

    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU()
    }

    if name.lower() not in activations:
        raise ValueError(f"Unsupported activations: {name}")
    
    return activations[name.lower()]