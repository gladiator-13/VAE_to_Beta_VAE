# From VAE to β-VAE: Implementing and Analysing KL Regularization and Latent Disentanglement

A from-scratch implementation of Variational Autoencoders (VAE) and β-VAE on MNIST, studying the effect of KL regularization strength and latent dimensionality on reconstruction quality and latent space structure.

Based on three papers read in sequence:
- Kingma & Welling (2013) — *Auto-Encoding Variational Bayes*
- Higgins et al. (2017) — *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*
- Burgess et al. (2018) — *Understanding Disentangling in β-VAE*

---

## Key Findings

**Latent Dimensionality Study (β = 1.0 fixed)**
- **The reconstruction elbow is at latent_dim = 10** — validation reconstruction loss drops sharply from dim=2 (155.45) to dim=10 (110.96), then flattens significantly through dim=32 (101.32), indicating dim=10 as the practical sweet spot for MNIST
- **Lower dimensions take longer to converge** — dim=5 required 60 epochs vs 40 for dim=10 and above, suggesting lower-dimensional spaces need more training to find compact representations
- **Higher dimensions show better val than train reconstruction** — at dim=10 and dim=20, validation reconstruction is meaningfully lower than training reconstruction, indicating that KL regularization prevents overfitting effectively at higher capacities

**KL Regularization Study (latent_dim = 2, with and without KL)**
- **KL term is essential for generative structure** — removing the KL term causes the latent space to expand uncontrollably (scale: 0–50 vs -4 to +4), with digit representations stretching into diagonal streaks rather than coherent clusters, confirming the model degenerates to a plain autoencoder without regularization

**β-VAE Sweep (in progress)**
- Results to be added after β sweep completion

---

## Results

### Latent Dimensionality Sweep (β = 1.0, 40–60 epochs)

| Latent Dim | Val Recon | Val KL | Val Loss | Epochs to Converge |
|------------|-----------|--------|----------|--------------------|
| 2          | 155.45    | 6.87   | 162.32   | 40                 |
| 3          | 134.67    | 9.81   | 144.49   | 40                 |
| 5          | 117.73    | 14.31  | 132.04   | 60                 |
| **10**     | **110.96**| **19.05**| **130.01** | **40**         |
| 20         | 105.00    | 24.60  | 129.60   | 40                 |
| 32         | 101.32    | 26.64  | 127.96   | 40                 |

Gains per step:
```
dim 2  → 5  :  -37.72 val recon  (large)
dim 5  → 10 :   -6.77 val recon  (meaningful)
dim 10 → 20 :   -5.96 val recon  (diminishing)
dim 20 → 32 :   -3.68 val recon  (marginal)
```

### β Sweep (latent_dim = 10, 40 epochs) — In Progress

| β    | Val Recon | Val KL | Val Loss | Latent Structure |
|------|-----------|--------|----------|-----------------|
| 0.5  | —         | —      | —        | —               |
| 1.0  | 110.96    | 19.05  | 130.01   | Baseline        |
| 2.0  | —         | —      | —        | —               |
| 4.0  | —         | —      | —        | —               |
| 8.0  | —         | —      | —        | —               |

---

## Visualizations

All visualizations are logged directly to Weights & Biases during training — no local plot folders needed.

### Latent Space — With vs Without KL Regularization (latent_dim = 2)

| With KL (VAE) | Without KL (Plain Autoencoder) |
|---|---|
| Centered around origin, scale ±4 | Uncontrolled, scale 0–50 |
| Smooth overlapping clusters | Stretched diagonal streaks |
| Generative — can sample from N(0,1) | Non-generative — space is arbitrary |

### Logged to W&B Per Run
- Reconstruction image grids (every 5 epochs)
- Random samples from prior N(0,1)
- 2D latent space scatter plot (digit-colored)
- Train/val loss curves (total, reconstruction, KL separately)

---

## Overview

This project implements VAE and β-VAE from scratch, deriving the ELBO loss and reparameterization trick directly from the original paper (Kingma & Welling, 2013), and studies two experimental axes:

1. **Latent dimensionality** — how much capacity does the latent space need?
2. **β regularization strength** — what is the tradeoff between reconstruction fidelity and latent space structure?

All experiments tracked with Weights & Biases. Best model checkpoints saved automatically per run.

---

## Project Structure

```
BETA_VAE_MNIST/
│
├── configs/
│   ├── dataset.py             # Dataset configuration
│   ├── evaluation_config.py   # Evaluation configuration
│   ├── model.py               # Model hyperparameters
│   ├── training.py            # Training configuration
│   └── wandb.py               # W&B configuration
│
├── data/
│   ├── MNIST/raw              # Auto-downloaded MNIST files
│   └── mnist.py               # MNIST DataModule
│
├── evaluation/
│   └── evaluator.py           # Evaluation pipeline
│
├── models/
│   ├── activations.py         # Custom activation functions
│   ├── output.py              # Model output dataclass
│   └── vae.py                 # VAE architecture
│
├── outputs/
│   └── checkpoints/           # Best model checkpoints per run
│
├── training/
│   ├── checkpoints.py         # Checkpoint saving logic
│   ├── losses.py              # ELBO loss (reconstruction + KL)
│   └── trainer.py             # Modular training loop
│
├── utils/
│   ├── checkpoints.py         # Checkpoint loading utilities
│   ├── logger.py              # Logging utilities
│   ├── seeds.py               # Reproducibility seeding
│   └── visualize.py           # Visualization utilities
│
├── visualization/
│   ├── latent.py              # 2D latent space visualization
│   ├── reconstruction.py      # Reconstruction visualization
│   └── sampling.py            # Random sample generation
│
├── wandb/                     # W&B local run cache
├── inference.py               # Inference pipeline
├── test_evaluate.py           # Evaluation script
├── train.py                   # Training entry point
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/<your-username>/beta-vae-mnist
cd beta-vae-mnist
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Train VAE

```bash
python train.py
```

### Evaluate

```bash
python test_evaluate.py
```

### Run Inference

```bash
python inference.py
```

---

## Experiment Tracking

All runs logged to Weights & Biases under project `vae-to-beta-vae`.

Each run tracks:
- `train/loss`, `val/loss` — total ELBO loss
- `train/reconstruction`, `val/reconstruction` — reconstruction term
- `train/kl`, `val/kl` — KL divergence term
- Reconstruction image grids every 5 epochs
- Random sample grids from prior N(0,1)
- 2D latent space visualization

Best model checkpoints saved automatically to `outputs/checkpoints/` per run.

---

## Stack

- Python
- PyTorch
- Weights & Biases
- torchvision
- matplotlib

---

## Roadmap

- [x] VAE implementation from scratch (ELBO, reparameterization trick)
- [x] MNIST DataModule
- [x] Modular trainer with validation loop
- [x] W&B experiment tracking
- [x] Automatic best model checkpointing
- [x] Reconstruction visualization
- [x] Random sample generation
- [x] 2D latent space visualization
- [x] Latent dimensionality sweep (dim = 2, 3, 5, 10, 20, 32)
- [ ] β-VAE implementation
- [ ] β sweep (β = 0.5, 1.0, 2.0, 4.0, 8.0)
- [ ] Latent interpolation
- [ ] Latent traversal
- [ ] Disentanglement analysis

---

## References

- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114
- Higgins, I., et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017
- Burgess, C., et al. (2018). *Understanding Disentangling in β-VAE*. arXiv:1804.03599

---

## License

For educational and research purposes.