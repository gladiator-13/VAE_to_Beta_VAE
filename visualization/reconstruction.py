import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

def plot_reconstructions(model, dataloader, device, num_images: int = 8) -> Figure:
    "Generate a figure comparing original and reconstructed images."
    IMAGE_WIDTH = 2
    IMAGE_HEIGHT = 2
    model.eval()

    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)

        images = images[:num_images]

        output = model(images, reconstruct=True)

        reconstructions = torch.sigmoid(output.x_logits)

        fig, axes = plt.subplots(
            2,
            num_images,
            figsize=(IMAGE_WIDTH * num_images, IMAGE_HEIGHT*2),
        )

        for i in range(num_images):
            axes[0, i].imshow(
                images[i].cpu().squeeze(),
                cmap="gray",
            )
            axes[0, i].axis("off")

            axes[1, i].imshow(
                reconstructions[i].cpu().squeeze(),
                cmap="gray",
            )
            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Reconstructed")

        plt.tight_layout()

        return fig