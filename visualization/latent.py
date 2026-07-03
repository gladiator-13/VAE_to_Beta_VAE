import torch
import matplotlib.pyplot as plt

def plot_latent_space(model, dataloader, device):
    """Visualize the latent space using the encoder
     means (mu) only when the latent dim is 2"""

    latent_vectors = []
    labels = []

    for images, target in dataloader:
        images = images.to(device)
        mu, _ = model.encode(images)
            
        latent_vectors.append(mu.cpu())
        labels.append(target)

    latent_vectors = torch.cat(latent_vectors).numpy()
    labels = torch.cat(labels).numpy()

    fig, ax = plt.subplots(figsize=(8,8))

    scatter = ax.scatter(
        latent_vectors[:, 0],
        latent_vectors[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.7
    )

    ax.set_title("Latent Space")
    ax.set_xlabel("Latent Dimnesion 1")
    ax.set_ylabel("Latent Dimnesion 2")

    plt.colorbar(scatter, ax=ax, label="Digit")

    return fig