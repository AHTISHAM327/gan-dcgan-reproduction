"""Datasets and DataLoaders for P12. Owner: Rija (taken over from Quratulain).

MNIST is read straight from the 4 official .gz files with our own Dataset class.
Splits follow the official GAN code: train = first 50,000 training images,
valid = last 10,000 training images, test = the 10,000 t10k images.
"""
import gzip
import os
import glob
import random
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

DRIVE_DATA = '/content/drive/MyDrive/gan-dcgan-reproduction-data/data'
MNIST_DIR = os.path.join(DRIVE_DATA, 'mnist')
CELEBA_ZIP = os.path.join(DRIVE_DATA, 'celeba-dataset.zip')


# ---------- MNIST ----------
def read_idx_images(path):
    """Read an MNIST image file (.gz, IDX format) into a uint8 array [N, 28, 28]."""
    with gzip.open(path, 'rb') as f:
        raw = f.read()
    magic, n, rows, cols = np.frombuffer(raw[:16], dtype='>i4')
    assert magic == 2051, f'{path} is not an IDX image file'
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)


def read_idx_labels(path):
    """Read an MNIST label file (.gz, IDX format) into an int64 array [N]."""
    with gzip.open(path, 'rb') as f:
        raw = f.read()
    magic, n = np.frombuffer(raw[:8], dtype='>i4')
    assert magic == 2049, f'{path} is not an IDX label file'
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


class MNISTFromIdx(Dataset):
    """Our own MNIST Dataset.
    split: 'train' (50,000) | 'valid' (10,000) | 'test' (10,000)
    scale: '01'  -> pixels in [0, 1]  (MLP GAN: sigmoid output, as in the GAN paper)
           'pm1' -> pixels in [-1, 1] (DCGAN: tanh output)"""

    def __init__(self, root=MNIST_DIR, split='train', scale='01'):
        assert split in ('train', 'valid', 'test') and scale in ('01', 'pm1')
        prefix = 't10k' if split == 'test' else 'train'
        imgs = read_idx_images(os.path.join(root, f'{prefix}-images-idx3-ubyte.gz'))
        labels = read_idx_labels(os.path.join(root, f'{prefix}-labels-idx1-ubyte.gz'))
        if split == 'train':
            imgs, labels = imgs[:50000], labels[:50000]
        elif split == 'valid':
            imgs, labels = imgs[50000:], labels[50000:]
        x = torch.from_numpy(imgs.copy()).float().div_(255.0).unsqueeze(1)   # [N, 1, 28, 28]
        self.x = x * 2 - 1 if scale == 'pm1' else x
        self.y = torch.from_numpy(labels.copy())
        self.split, self.scale = split, scale

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ---------- CelebA ----------
def prepare_celeba(zip_path=CELEBA_ZIP, out_dir='/content/celeba'):
    """Copy the CelebA zip from Drive to Colab's local disk and unzip it (once per session).
    Returns (img_dir, partition_csv)."""
    if not os.path.exists(out_dir):
        shutil.copy(zip_path, '/content/celeba.zip')
        subprocess.run(['unzip', '-q', '/content/celeba.zip', '-d', out_dir], check=True)
    img_dir = os.path.dirname(next(glob.iglob(f'{out_dir}/**/*.jpg', recursive=True)))
    part_csv = glob.glob(f'{out_dir}/**/list_eval_partition.csv', recursive=True)[0]
    return img_dir, part_csv


class CelebAFaces(Dataset):
    """Our own CelebA Dataset for one official split (0 = train, 1 = valid, 2 = test).
    Returns (image, -1) so every dataset gives (image, label); CelebA has no class labels."""

    def __init__(self, img_dir, partition_csv, split=0, image_size=64):
        df = pd.read_csv(partition_csv)
        self.files = df.loc[df['partition'] == split, 'image_id'].tolist()
        self.img_dir = img_dir
        self.transform = T.Compose([T.CenterCrop(178), T.Resize(image_size), T.ToTensor(),
                                    T.Normalize([0.5] * 3, [0.5] * 3)])   # -> [-1, 1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.files[i])).convert('RGB')
        return self.transform(img), -1


# ---------- Loaders and checks ----------
def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size=128, shuffle=True, seed=42, num_workers=2, drop_last=True):
    """DataLoader with a fixed seed, so the batch order is the same every run."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, worker_init_fn=_seed_worker, generator=g,
                      pin_memory=torch.cuda.is_available())


def compute_norm_stats(dataset, max_items=None):
    """Mean and std per channel, computed on the dataset you pass (use the TRAIN split only)."""
    n = len(dataset) if max_items is None else min(max_items, len(dataset))
    total, total_sq, count = 0.0, 0.0, 0
    for i in range(n):
        x = dataset[i][0].double()
        total = total + x.sum(dim=(1, 2))
        total_sq = total_sq + (x ** 2).sum(dim=(1, 2))
        count += x.shape[1] * x.shape[2]
    mean = total / count
    std = (total_sq / count - mean ** 2).sqrt()
    return mean.float(), std.float()


def check_batch(loader, expected_shape, expected_range):
    """Print one batch's shape, dtype and value range, and stop with an error if they are wrong."""
    x, y = next(iter(loader))
    lo, hi = x.min().item(), x.max().item()
    print(f'images: {tuple(x.shape)} {x.dtype} | min {lo:.2f} max {hi:.2f}')
    print(f'labels: {tuple(y.shape)} {y.dtype} | first 10: {y[:10].tolist()}')
    assert tuple(x.shape) == tuple(expected_shape), f'shape {tuple(x.shape)} != {expected_shape}'
    assert x.dtype == torch.float32, 'images must be float32'
    assert expected_range[0] - 1e-6 <= lo and hi <= expected_range[1] + 1e-6, 'values out of range'
    print('OK: shape, dtype and range are correct')
    return x, y
