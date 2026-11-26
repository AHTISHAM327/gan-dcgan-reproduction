"""Training loop for P12. Owner: Rija.

One D step on every batch, one G step every `d_steps` batches (paper: k = 1).
Mixed precision on the GPU, fixed seeds, and a checkpoint saved after every epoch so
training can continue after Colab disconnects.
Optional schedules from the official GAN code (mnist.yaml): learning rate divided by 1.000004
after every batch (min 1e-6), and momentum raised from 0.5 to 0.7 over the first 250 epochs.
"""
import math
import os
import random
import time

import numpy as np
import torch
from torchvision.utils import save_image

from src.losses import d_loss_fn, g_loss_fn

HISTORY_KEYS = ['epoch', 'd_loss', 'g_loss', 'd_real', 'd_fake',
                'val_d_loss', 'val_d_real', 'val_d_fake', 'time_s']


def set_seed(seed=42):
    """Same seed -> same weights, same noise, same batch order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_noise(n, z_dim, kind='uniform', device='cpu'):
    """Uniform noise in [-sqrt(3), sqrt(3)] (official GAN code) or standard normal."""
    if kind == 'uniform':
        return (torch.rand(n, z_dim, device=device) * 2 - 1) * math.sqrt(3)
    return torch.randn(n, z_dim, device=device)


def _images(batch):
    return batch[0] if isinstance(batch, (list, tuple)) else batch


class GANTrainer:
    def __init__(self, G, D, z_dim=100, noise='uniform', optimizer='adam', lr=2e-4, betas=(0.5, 0.999), momentum=0.5,
                 d_steps=1, real_label=1.0, amp=True, ckpt_path=None, sample_dir=None,
                 sample_range=(0, 1), seed=42, device=None,
                 lr_decay=None, min_lr=1e-6, final_momentum=None, momentum_saturate=250, snapshot_every=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.G, self.D = G.to(self.device), D.to(self.device)
        self.z_dim, self.noise, self.d_steps, self.real_label = z_dim, noise, d_steps, real_label
        if optimizer == 'sgd':      # the GAN paper's optimizer: SGD with momentum
            self.opt_g = torch.optim.SGD(self.G.parameters(), lr=lr, momentum=momentum)
            self.opt_d = torch.optim.SGD(self.D.parameters(), lr=lr, momentum=momentum)
        else:                       # the DCGAN paper's optimizer: Adam
            self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas)
            self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        self.use_amp = amp and self.device.type == 'cuda'
        self.scaler_g = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scaler_d = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.ckpt_path, self.sample_dir, self.sample_range = ckpt_path, sample_dir, sample_range
        g = torch.Generator().manual_seed(seed)      # fixed noise -> the same 64 samples every epoch
        if noise == 'uniform':
            self.fixed_z = ((torch.rand(64, z_dim, generator=g) * 2 - 1) * math.sqrt(3)).to(self.device)
        else:
            self.fixed_z = torch.randn(64, z_dim, generator=g).to(self.device)
        self.history = {k: [] for k in HISTORY_KEYS}
        self.epoch = 0
        self.lr_decay, self.min_lr = lr_decay, min_lr
        self.momentum0, self.final_momentum, self.momentum_saturate = momentum, final_momentum, momentum_saturate
        self.snapshot_every = snapshot_every

    def _set_momentum(self, epoch):
        """Linear ramp from the start momentum to final_momentum, reached at epoch momentum_saturate."""
        if self.final_momentum is None:
            return
        a = min(1.0, max(0.0, (epoch - 1) / max(1, self.momentum_saturate - 1)))
        m = self.momentum0 + a * (self.final_momentum - self.momentum0)
        for opt in (self.opt_d, self.opt_g):
            for group in opt.param_groups:
                if 'momentum' in group:
                    group['momentum'] = m

    def _decay_lr(self):
        if self.lr_decay is None:
            return
        for opt in (self.opt_d, self.opt_g):
            for group in opt.param_groups:
                group['lr'] = max(group['lr'] / self.lr_decay, self.min_lr)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp)

    def _noise(self, n):
        return sample_noise(n, self.z_dim, self.noise, self.device)

    def train_epoch(self, loader):
        self.G.train(); self.D.train()
        sums = {'d_loss': 0.0, 'd_real': 0.0, 'd_fake': 0.0, 'g_loss': 0.0}
        n_d = n_g = 0
        for i, batch in enumerate(loader):
            real = _images(batch).to(self.device, non_blocking=True)
            b = real.shape[0]
            # ---- D step: maximise log D(x) + log(1 - D(G(z))) ----
            with self._autocast():
                fake = self.G(self._noise(b))
                real_logits = self.D(real)
                fake_logits = self.D(fake.detach())
            d_loss, _, _ = d_loss_fn(real_logits.float(), fake_logits.float(), self.real_label)
            self.opt_d.zero_grad(set_to_none=True)
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
            sums['d_loss'] += d_loss.item()
            sums['d_real'] += torch.sigmoid(real_logits.float()).mean().item()
            sums['d_fake'] += torch.sigmoid(fake_logits.float()).mean().item()
            n_d += 1
            # ---- G step (every d_steps batches): maximise log D(G(z)) ----
            if (i + 1) % self.d_steps == 0:
                with self._autocast():
                    g_logits = self.D(self.G(self._noise(b)))
                g_loss = g_loss_fn(g_logits.float())
                self.opt_g.zero_grad(set_to_none=True)
                self.scaler_g.scale(g_loss).backward()
                self.scaler_g.step(self.opt_g)
                self.scaler_g.update()
                sums['g_loss'] += g_loss.item()
                n_g += 1
            self._decay_lr()
        return {'d_loss': sums['d_loss'] / n_d, 'd_real': sums['d_real'] / n_d,
                'd_fake': sums['d_fake'] / n_d, 'g_loss': sums['g_loss'] / max(n_g, 1)}

    @torch.no_grad()
    def validate(self, loader):
        """D on held-out real images (and fresh fakes), in eval mode. If D(x) on validation
        falls far below D(x) on training, D is memorising the training set."""
        self.G.eval(); self.D.eval()
        tot = {'val_d_loss': 0.0, 'val_d_real': 0.0, 'val_d_fake': 0.0}
        n = 0
        for batch in loader:
            real = _images(batch).to(self.device)
            with self._autocast():
                rl = self.D(real)
                fl = self.D(self.G(self._noise(real.shape[0])))
            d_loss, _, _ = d_loss_fn(rl.float(), fl.float())
            tot['val_d_loss'] += d_loss.item()
            tot['val_d_real'] += torch.sigmoid(rl.float()).mean().item()
            tot['val_d_fake'] += torch.sigmoid(fl.float()).mean().item()
            n += 1
        return {k: v / n for k, v in tot.items()}

    def save_checkpoint(self):
        state = {'epoch': self.epoch, 'G': self.G.state_dict(), 'D': self.D.state_dict(),
                 'opt_g': self.opt_g.state_dict(), 'opt_d': self.opt_d.state_dict(),
                 'scaler_g': self.scaler_g.state_dict(), 'scaler_d': self.scaler_d.state_dict(),
                 'history': self.history, 'fixed_z': self.fixed_z.cpu(),
                 'rng_cpu': torch.get_rng_state(),
                 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
        tmp = self.ckpt_path + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)        # write fully, then swap: a half-written file never replaces a good one

    def load_checkpoint(self):
        """Load the checkpoint if it exists. Returns the epoch to start from."""
        if not (self.ckpt_path and os.path.exists(self.ckpt_path)):
            return 1
        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.G.load_state_dict(state['G']); self.D.load_state_dict(state['D'])
        self.opt_g.load_state_dict(state['opt_g']); self.opt_d.load_state_dict(state['opt_d'])
        self.scaler_g.load_state_dict(state['scaler_g']); self.scaler_d.load_state_dict(state['scaler_d'])
        self.history, self.epoch = state['history'], state['epoch']
        self.fixed_z = state['fixed_z'].to(self.device)
        torch.set_rng_state(state['rng_cpu'].cpu())

if state['rng_cuda'] is not None and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(
        [s.cpu() for s in state['rng_cuda']]
    )
        print(f'Resumed from {self.ckpt_path} (finished epoch {self.epoch})')
        return self.epoch + 1

    @torch.no_grad()
    def save_samples(self, epoch):
        if not self.sample_dir:
            return
        os.makedirs(self.sample_dir, exist_ok=True)
        self.G.eval()
        imgs = self.G(self.fixed_z).float()
        save_image(imgs, os.path.join(self.sample_dir, f'epoch_{epoch:03d}.png'), nrow=8,
                   normalize=True, value_range=self.sample_range)

    def fit(self, train_loader, valid_loader=None, epochs=50, sample_every=5, resume=True):
        start = self.load_checkpoint() if resume else 1
        for epoch in range(start, epochs + 1):
            t0 = time.time()
            self._set_momentum(epoch)
            tr = self.train_epoch(train_loader)
            va = self.validate(valid_loader) if valid_loader is not None else \
                {'val_d_loss': float('nan'), 'val_d_real': float('nan'), 'val_d_fake': float('nan')}
            self.epoch = epoch
            row = {'epoch': epoch, **tr, **va, 'time_s': time.time() - t0}
            for k in HISTORY_KEYS:
                self.history[k].append(row[k])
            print(f"epoch {epoch:3d} | D loss {tr['d_loss']:.3f} | G loss {tr['g_loss']:.3f} | "
                  f"D(x) {tr['d_real']:.3f} | D(G(z)) {tr['d_fake']:.3f} | "
                  f"val D(x) {va['val_d_real']:.3f} | {row['time_s']:.0f}s")
            if self.ckpt_path:
                self.save_checkpoint()
                if self.snapshot_every and epoch % self.snapshot_every == 0:
                    stem = os.path.splitext(self.ckpt_path)[0]
                    torch.save(self.G.state_dict(), f'{stem}_G_epoch{epoch:03d}.pt')
            if epoch % sample_every == 0 or epoch == epochs:
                self.save_samples(epoch)
        return self.history


def overfit_one_batch_D(D, real, fake, steps=300, lr=1e-3, device=None):
    """Overfit test: D alone on ONE fixed batch of real images and ONE fixed batch of fakes.
    If the code is correct, the D loss goes close to 0. Returns the list of losses."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    D = D.to(device).train()
    real, fake = real.to(device), fake.to(device).detach()
    opt = torch.optim.Adam(D.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        loss, _, _ = d_loss_fn(D(real), D(fake))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == steps - 1:
            print(f'step {step:3d} | D loss {loss.item():.4f}')
    return losses
