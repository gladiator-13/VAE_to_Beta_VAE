import torch
import matplotlib.pyplot as plt


def plot_latent_traversal(model, device, latent_range=(-3, 3), steps=11):
    values = torch.linspace(
        latent_range[0],
        latent_range[1],
        steps,
    )

    latent_vectors = []

    for y in values:
        for x in values:
            latent_vectors.append([x.item(), y.item()])

    z = torch.tensor(
        latent_vectors,
        dtype=torch.float32,
        device=device,
    )

    logits = model.decode(z)
    samples = torch.sigmoid(logits).cpu()

    fig, axes = plt.subplots(
        steps,
        steps,
        figsize=(10, 10),
    )

    fig.suptitle("2D Latent Traversal", fontsize=16)

    index = 0

    for row in range(steps):
        for col in range(steps):

            axes[row, col].imshow(
                samples[index].reshape(28, 28),
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[row, col].axis("off")

            index += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig