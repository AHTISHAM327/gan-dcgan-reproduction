"""Generator and Discriminator models for P12. Owner: Ahtisham.

MLP GAN = the GAN paper's MNIST model (layer sizes from the official code, goodfeli/adversarial, mnist.yaml):
  G: z(100) -> 1200 ReLU -> 1200 ReLU -> 784 sigmoid
  D: maxout(240 units x 5 pieces) -> maxout(240 x 5) -> 1, with dropout (0.2 on the input, 0.5 after)
DCGAN = Radford et al. 2016, Figure 1 shapes: z(100) -> 4x4x1024 -> 8x8x512 -> 16x16x256 -> 32x32x128 -> 64x64x3.
D returns LOGITS (no sigmoid); the loss in losses.py applies the sigmoid safely.
"""
import torch
import torch.nn as nn


# ---------------- MLP GAN (reproduction, MNIST) ----------------
class Maxout(nn.Module):
    """Maxout layer: a Linear layer gives units*pieces values; keep the max of each group of `pieces`."""

    def __init__(self, in_features, units, pieces):
        super().__init__()
        self.units, self.pieces = units, pieces
        self.linear = nn.Linear(in_features, units * pieces)

    def forward(self, x):
        out = self.linear(x)                                   # [B, units*pieces]
        return out.view(-1, self.units, self.pieces).max(dim=2).values   # [B, units]


class MLPGenerator(nn.Module):
    def __init__(self, z_dim=100, hidden=1200, img_shape=(1, 28, 28)):
        super().__init__()
        self.z_dim, self.img_shape = z_dim, img_shape
        out_dim = img_shape[0] * img_shape[1] * img_shape[2]
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        assert z.dim() == 2 and z.shape[1] == self.z_dim, f'z must be [B, {self.z_dim}], got {tuple(z.shape)}'
        x = self.net(z).view(-1, *self.img_shape)               # [B, 1, 28, 28], values in (0, 1)
        return x


class MLPDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), units=240, pieces=5):
        super().__init__()
        self.img_shape = img_shape
        in_dim = img_shape[0] * img_shape[1] * img_shape[2]
        self.net = nn.Sequential(
            nn.Flatten(),                                       # [B, 784]
            nn.Dropout(0.2), Maxout(in_dim, units, pieces),     # [B, 240]
            nn.Dropout(0.5), Maxout(units, units, pieces),      # [B, 240]
            nn.Dropout(0.5), nn.Linear(units, 1),               # [B, 1]
        )

    def forward(self, x):
        assert tuple(x.shape[1:]) == tuple(self.img_shape), f'x must be [B, {self.img_shape}], got {tuple(x.shape)}'
        return self.net(x).view(-1)                             # [B] logits


def init_mlp_like_paper(G, D, pixel_mean=None):
    """Initialise like the official code: weights uniform in [-0.05, 0.05] (G) and [-0.005, 0.005] (D), biases 0.
    Optional: G's output bias = logit of the mean training image (the official code does this)."""
    for model, r in ((G, 0.05), (D, 0.005)):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -r, r)
                nn.init.zeros_(m.bias)
    if pixel_mean is not None:
        p = pixel_mean.reshape(-1).clamp(1e-3, 1 - 1e-3)
        G.net[4].bias.data.copy_(torch.log(p / (1 - p)))


# ---------------- DCGAN (new dataset, CelebA 64x64) ----------------
class DCGANGenerator(nn.Module):
    """Shapes follow DCGAN Figure 1. Kernel 4, stride 2, padding 1 doubles the size each block (our choice of kernel)."""

    def __init__(self, z_dim=100, ngf=128, channels=3):
        super().__init__()
        self.z_dim = z_dim

        def up(cin, cout):
            return [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),  # 4x4x1024
            *up(ngf * 8, ngf * 4),                                                                           # 8x8x512
            *up(ngf * 4, ngf * 2),                                                                           # 16x16x256
            *up(ngf * 2, ngf),                                                                               # 32x32x128
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False), nn.Tanh(),                              # 64x64x3
        )

    def forward(self, z):
        assert z.dim() == 2 and z.shape[1] == self.z_dim, f'z must be [B, {self.z_dim}], got {tuple(z.shape)}'
        return self.net(z.view(-1, self.z_dim, 1, 1))


class DCGANDiscriminator(nn.Module):
    """Mirror of the generator (our choice: the paper does not list D's sizes).
    No batch norm on the first layer, LeakyReLU 0.2 everywhere (paper guidelines)."""

    def __init__(self, ndf=128, channels=3):
        super().__init__()

        def down(cin, cout):
            return [nn.Conv2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, True)]

        self.net = nn.Sequential(
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),   # 32x32x128
            *down(ndf, ndf * 2),                                                    # 16x16x256
            *down(ndf * 2, ndf * 4),                                                # 8x8x512
            *down(ndf * 4, ndf * 8),                                                # 4x4x1024
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),                             # 1x1x1
        )

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1:] == (3, 64, 64), f'x must be [B, 3, 64, 64], got {tuple(x.shape)}'
        return self.net(x).view(-1)                                                 # [B] logits


def dcgan_init(model):
    """DCGAN paper init: weights from N(0, 0.02); batch-norm scale from N(1, 0.02), shift 0."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)


# ---------------- helpers for the diagram and the hand count ----------------
def count_params(module):
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_layer_shapes(model, example_input):
    """Run one input through the model and print the output shape after every layer
    (use this table for the architecture diagram). For a Maxout layer you see its inner
    Linear (units x pieces) first, then the Maxout output (units)."""
    rows = []
    hooks = [m.register_forward_hook(lambda m, i, o, n=name: rows.append((n, type(m).__name__, tuple(o.shape))))
             for name, m in model.named_modules() if len(list(m.children())) == 0 or isinstance(m, Maxout)]
    model.eval()
    with torch.no_grad():
        model(example_input)
    for h in hooks:
        h.remove()
    print(f'{"layer":<14}{"type":<20}{"output shape"}')
    for name, kind, shape in rows:
        print(f'{name:<14}{kind:<20}{shape}')
