import math
import matplotlib.pyplot as plt 
import torch

def plot_random_samples(model, device, num_samples: int=16):
    "Generate Random samples for the VAE's prior"

    z = torch.randn(
        num_samples,
        model.config.latent_dim,
        device=device
    )

    x_logits = model.decode(z)
    samples = torch.sigmoid(x_logits)

    #Create grid
    grid_size = math.ceil(math.sqrt(num_samples))

    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize = (2*grid_size, 2*grid_size)
    )

    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(
            samples[i].cpu().squeeze(),
            cmap="gray"
        )
        axes[i].axis("off")

    #hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")
        
    plt.suptitle("Ransom Samples")
    plt.tight_layout()

    return fig



