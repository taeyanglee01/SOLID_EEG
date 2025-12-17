import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from termcolor import colored
import ssl
import os
from glob import glob
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context


class customDataset(Dataset):
    def __init__(self, folder, transform, exts=['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.paths = [p for ext in exts for p in glob(os.path.join(folder, f'*.{ext}'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        img = self.transform(img)  # shape (3, H, W)

        # Generate sparse mask (10% of pixels kept)
        C, H, W = img.shape
        total_pixels = H * W
        num_known = int(0.1 * total_pixels)

        mask = torch.zeros(H * W)
        idx = torch.randperm(H * W)[:num_known]
        mask[idx] = 1
        mask = mask.view(1, H, W)  # (1, H, W)
        mask = mask.repeat(C, 1, 1)  # (C, H, W)

        sparse_img = img * mask  # masked version of image

        # Return as tuple to keep trainer.py unchanged
        return torch.cat([img, sparse_img, mask], dim=0)  # shape (9, H, W)



import numpy as np
import torch
from torch.utils.data import Dataset

class NavierStokesCondDDPMDataset(Dataset):
    """
    Returns tensor shaped (9, H, W) = concat([y, x_sparse, mask]) where each block has 3 channels
    (repeated from the single-channel field) so your trainer can stay unchanged:
      - [:3]   -> target y (next step)
      - [3:6]  -> sparse x (current step masked)
      - [6:9]  -> mask (0/1)
    Masks are persistent per (item_idx, t) via SeedSequence (order/epoch/worker independent).
    """
    def __init__(
        self,
        memmap,                   # indexable: memmap[item_idx] -> (T, H, W)
        index_pairs,              # np.ndarray[N,2] of (item_idx, t)
        sparsity=0.2,
        seed=42,
        normalize=True,
        mean=0.0,
        std=2.4036,
        use_nan=False,            # if True, drop pixels -> NaN; else -> 0
        persist_masks=True
    ):
        self.mem = memmap
        self.index_pairs = index_pairs
        self.sparsity = float(sparsity)
        self.seed = int(seed) if seed is not None else None
        self.normalize = bool(normalize)
        self.mean = float(mean)
        self.std = float(std)
        self.use_nan = bool(use_nan)
        self.persist_masks = bool(persist_masks)
        self._mask_cache = {}

    def __len__(self):
        return len(self.index_pairs)

    @staticmethod
    def _z_norm(arr, mean, std):
        return (arr - mean) / std

    def _per_image_mask(self, item_idx, t, shape):
        if self.sparsity >= 1.0:
            return None
        key = (int(item_idx), int(t), int(shape[0]), int(shape[1]))
        if self.persist_masks and key in self._mask_cache:
            return self._mask_cache[key]

        # Overflow-free, deterministic seeding by (seed, item_idx, t)
        if self.seed is None:
            ss = np.random.SeedSequence([int(item_idx), int(t)])
        else:
            ss = np.random.SeedSequence([int(self.seed), int(item_idx), int(t)])
        rng = np.random.default_rng(np.random.PCG64(ss))

        mask = (rng.random(shape) < self.sparsity).astype(np.float32)  # (H, W), 0/1
        if self.persist_masks:
            self._mask_cache[key] = mask
        return mask

    def __getitem__(self, i):
        item_idx, t = self.index_pairs[i]
        seq = self.mem[item_idx]                  # (T, H, W)
        x  = np.asarray(seq[t],   dtype=np.float32)   # (H, W)
        y  = np.asarray(seq[t+1], dtype=np.float32)   # (H, W)

        if self.normalize:
            x = self._z_norm(x, self.mean, self.std)
            y = self._z_norm(y, self.mean, self.std)

        # persistent per-image mask on x
        if self.sparsity < 1.0:
            m = self._per_image_mask(item_idx, t, x.shape)  # (H, W), float 0/1
            if self.use_nan:
                x_sparse = np.where(m > 0, x, np.nan)
            else:
                x_sparse = x * m
        else:
            m = np.ones_like(x, dtype=np.float32)
            x_sparse = x

        # to torch and repeat to 3 channels to match trainer's CIFAR interface
        y_t        = torch.from_numpy(y[None, ...])        # (1,H,W)
        x_sparse_t = torch.from_numpy(x_sparse[None, ...]) # (1,H,W)
        m_t        = torch.from_numpy(m[None, ...])        # (1,H,W)

        y_t        = y_t.repeat(3, 1, 1)        # (3,H,W)
        x_sparse_t = x_sparse_t.repeat(3, 1, 1) # (3,H,W)
        m_t        = m_t.repeat(3, 1, 1)        # (3,H,W)

        sample = torch.cat([y_t, x_sparse_t, m_t], dim=0)  # (9,H,W)
        return sample



# def dataset_wrapper(dataset, image_size, augment_horizontal_flip=True, info_color='green', min1to1=True):
#     transform = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.RandomHorizontalFlip() if augment_horizontal_flip else torch.nn.Identity(),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),  # turn into torch Tensor of shape CHW, 0 ~ 1
#         transforms.Lambda(lambda x: ((x * 2) - 1)) if min1to1 else torch.nn.Identity()# -1 ~ 1
#     ])
#     if os.path.isdir(dataset):
#         print(colored('Loading local file directory', info_color))
#         dataSet = customDataset(dataset, transform)
#         print(colored('Successfully loaded {} images!'.format(len(dataSet)), info_color))
#         return dataSet
#     else:
#         dataset = dataset.lower()
#         assert dataset in ['cifar10']
#         print(colored('Loading {} dataset'.format(dataset), info_color))
#         if dataset == 'cifar10':
#             trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#             testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#             fullset = torch.utils.data.ConcatDataset([trainset, testset])
#             return fullset
#         elif dataset =='cifar10_test':
#             testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#             return tesetset


def dataset_wrapper(dataset, image_size, augment_horizontal_flip=True, info_color='green', min1to1=True, **kwargs):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip() if augment_horizontal_flip else torch.nn.Identity(),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: ((x * 2) - 1)) if min1to1 else torch.nn.Identity()
    ])
    if os.path.isdir(dataset):
        print(colored('Loading local file directory', info_color))
        dataSet = customDataset(dataset, transform)
        print(colored('Successfully loaded {} images!'.format(len(dataSet)), info_color))
        return dataSet
    else:
        dataset = dataset.lower()
        if dataset == 'navierstokes':
            print(colored('Loading Navier–Stokes memmap dataset', info_color))
            # required kwargs: memmap, index_pairs
            memmap = kwargs['memmap']
            index_pairs = kwargs['index_pairs']
            ds = NavierStokesCondDDPMDataset(
                memmap=memmap,
                index_pairs=index_pairs,
                sparsity=kwargs.get('sparsity', 0.2),
                seed=kwargs.get('seed', 42),
                normalize=kwargs.get('normalize', True),
                mean=kwargs.get('mean', 0.0),
                std=kwargs.get('std', 2.4036),
                use_nan=kwargs.get('use_nan', False),
                persist_masks=True
            )
            # no torchvision transform here; NS is already numeric (64x64)
            return ds

        assert dataset in ['cifar10', 'cifar10_test']
        print(colored('Loading {} dataset'.format(dataset), info_color))
        if dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            fullset  = torch.utils.data.ConcatDataset([trainset, testset])
            return fullset
        elif dataset == 'cifar10_test':
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            return testset
