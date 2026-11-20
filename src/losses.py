"""Adversarial losses written from the GAN equation (Goodfellow et al., 2014).
Owner: Ahtisham ("Beyond the lecture": our own loss, checked against PyTorch).

D returns logits l (numbers before the sigmoid), so D(x) = sigmoid(l).
  -log(sigmoid(l))     = softplus(-l)
  -log(1 - sigmoid(l)) = softplus(l)
softplus is numerically stable (it never computes log(0)).
"""
import torch
import torch.nn.functional as F


def d_loss_fn(real_logits, fake_logits, real_label=1.0):
    """Discriminator loss = -[ y*log D(x) + (1-y)*log(1-D(x)) ] - log(1 - D(G(z))).
    y = real_label (1.0 normally; 0.9 for the label-smoothing ablation).
    Returns (total, real_part, fake_part). At the start each part is about ln 2 = 0.693."""
    y = real_label
    real_part = (y * F.softplus(-real_logits) + (1 - y) * F.softplus(real_logits)).mean()
    fake_part = F.softplus(fake_logits).mean()
    return real_part + fake_part, real_part, fake_part


def g_loss_fn(fake_logits):
    """Non-saturating generator loss from the paper: G maximises log D(G(z)),
    so we minimise -log D(G(z)) = softplus(-l)."""
    return F.softplus(-fake_logits).mean()


def check_against_pytorch(n=10000, real_label=1.0, seed=0):
    """Compare our losses with PyTorch's built-in BCEWithLogitsLoss on random logits.
    Returns the largest absolute difference (should be < 1e-6)."""
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(n, generator=g) * 4
    fake = torch.randn(n, generator=g) * 4
    bce = torch.nn.BCEWithLogitsLoss()
    ours_d, ours_r, ours_f = d_loss_fn(real, fake, real_label)
    ref_r = bce(real, torch.full((n,), real_label))
    ref_f = bce(fake, torch.zeros(n))
    ours_g, ref_g = g_loss_fn(fake), bce(fake, torch.ones(n))
    diffs = {
        'D real part': (ours_r - ref_r).abs().item(),
        'D fake part': (ours_f - ref_f).abs().item(),
        'D total': (ours_d - (ref_r + ref_f)).abs().item(),
        'G loss': (ours_g - ref_g).abs().item(),
    }
    for k, v in diffs.items():
        print(f'{k:12s} difference = {v:.2e}')
    return max(diffs.values())
