"""Plots for P12: batches with labels, training curves, sample grids. Owner: Rija (taken over from Quratulain)."""
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def show_batch_with_labels(x, y, n=16, path=None, value_range=(0, 1), title='One batch'):
    """Plot the first n images of a batch, each with its label as a title."""
    n = min(n, len(x))
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    lo, hi = value_range
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < n:
            img = ((x[i] - lo) / (hi - lo)).clamp(0, 1).permute(1, 2, 0).squeeze(-1).cpu()
            ax.imshow(img, cmap='gray' if img.dim() == 2 else None)
            ax.set_title(str(int(y[i])), fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_curves(history, path=None, title='Baseline run: MLP GAN on MNIST'):
    """Two panels: losses (left) and D outputs (right), per epoch.
    Train and validation are both shown, as the course checklist asks."""
    ep = history['epoch']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, history['d_loss'], label='D loss (train)')
    a1.plot(ep, history['g_loss'], label='G loss (train)')
    a1.plot(ep, history['val_d_loss'], '--', label='D loss (validation)')
    a1.set_xlabel('epoch'); a1.set_ylabel('loss'); a1.set_title('Losses'); a1.legend()
    a2.plot(ep, history['d_real'], label='D(x) real (train)')
    a2.plot(ep, history['d_fake'], label='D(G(z)) fake (train)')
    a2.plot(ep, history['val_d_real'], '--', label='D(x) real (validation)')
    a2.axhline(0.5, color='gray', linestyle=':', label='0.5 = equilibrium')
    a2.set_ylim(0, 1); a2.set_xlabel('epoch'); a2.set_ylabel('mean D output')
    a2.set_title('Discriminator outputs'); a2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()


def show_grid(images, title='', path=None, value_range=(0, 1), nrow=8):
    """Show (and optionally save) a grid of images, e.g. generated samples."""
    grid = make_grid(images[:64].float().cpu(), nrow=nrow, normalize=True, value_range=value_range)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off'); plt.title(title)
    if path:
        plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
