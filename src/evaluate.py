"""Parzen-window log-likelihood on MNIST (GAN paper, Table 1). Owner: Ahtisham.

Same procedure as the official GAN code (goodfeli/adversarial, parzen_ll.py):
1. draw 10,000 samples from G,
2. put a Gaussian of width sigma on every sample,
3. pick sigma from 10 values between 0.1 and 1 (log scale) using the first 1,000 validation images,
4. report the mean log-likelihood of the 10,000 test images, +- its standard error.
"""
import math

import numpy as np
import torch

from src.train import sample_noise


@torch.no_grad()
def generate_samples(G, n=10000, z_dim=100, noise='uniform', seed=0, device=None, batch=1000):
    """n images from G, flattened to [n, 784], pixel values in [0, 1]."""
    device = device or next(G.parameters()).device
    G.eval()
    torch.manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        z = sample_noise(min(batch, n - i), z_dim, noise, device)
        out.append(G(z).float().reshape(z.shape[0], -1))
    return torch.cat(out)


def _sq_dists(x, mu):
    """Squared distance from every row of x to every row of mu: [len(x), len(mu)]."""
    d = (x * x).sum(1, keepdim=True) - 2 * x @ mu.T + (mu * mu).sum(1)
    return d.clamp_min(0)


@torch.no_grad()
def parzen_ll(x, mu, sigma, batch=500):
    """Log-likelihood of each row of x under a Gaussian Parzen window centred on the rows of mu."""
    n, dim = mu.shape
    log_z = dim * math.log(sigma * math.sqrt(2 * math.pi))
    out = []
    for i in range(0, len(x), batch):
        d = _sq_dists(x[i:i + batch], mu)
        out.append(torch.logsumexp(-0.5 * d / sigma ** 2, dim=1) - math.log(n) - log_z)
    return torch.cat(out)


def choose_sigma(mu, valid_x, sigmas=None):
    """Try each sigma on the validation images and keep the one with the highest mean log-likelihood."""
    sigmas = np.logspace(-1, 0, 10) if sigmas is None else sigmas
    scores = []
    for s in sigmas:
        ll = parzen_ll(valid_x, mu, float(s)).mean().item()
        scores.append(ll)
        print(f'sigma {s:.3f} -> validation log-likelihood {ll:8.1f}')
    best = float(sigmas[int(np.argmax(scores))])
    return best, scores


def parzen_evaluate(G, valid_x, test_x, n_samples=10000, seed=0, device=None):
    """Full evaluation. valid_x / test_x: tensors [N, 784] in [0, 1]. Returns (mean, standard error, sigma)."""
    device = device or next(G.parameters()).device
    mu = generate_samples(G, n_samples, seed=seed, device=device)
    sigma, _ = choose_sigma(mu, valid_x.to(device))
    ll = parzen_ll(test_x.to(device), mu, sigma)
    mean, se = ll.mean().item(), (ll.std() / math.sqrt(len(ll))).item()
    print(f'\nchosen sigma = {sigma:.3f}')
    print(f'Parzen log-likelihood on {len(ll):,} test images = {mean:.1f} +- {se:.1f}   (paper: 225 +- 2)')
    return mean, se, sigma
