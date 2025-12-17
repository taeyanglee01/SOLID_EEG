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



# --- Navier–Stokes: next-step target-only dataset for DDPM ---

# src/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

# class NavierStokesNextStepDDPM(Dataset):
#     """
#     Navier–Stokes next-step pairs, single-channel throughout.

#     Returns:
#       - y_{t+1}: FloatTensor (1, H, W)  # single channel
#       - sample_id: int64 stable ID = item_idx*T + t  (useful for persistent masks)

#     Notes:
#       * Normalization is applied per your mean/std if requested.
#       * to_unit_range=True applies tanh ([-1,1]) after z-norm (optional).
#     """
#     def __init__(self, memmap, index_pairs,
#                  normalize=False, mean=0.0, std=1.0, to_unit_range=False,
#                  max_samples=None, verbose=True):
#         index_pairs = np.asarray(index_pairs, dtype=np.int64)
#         if max_samples is not None:
#             index_pairs = index_pairs[:max_samples]
#         self.mem = memmap
#         self.index_pairs = index_pairs
#         self.normalize = bool(normalize)
#         self.mean = float(mean)
#         self.std = float(std)
#         self.to_unit_range = bool(to_unit_range)

#         # Assume fixed T for chosen items; keep for sample_id packing
#         it0 = int(self.index_pairs[0, 0])
#         self.T = int(self.mem[it0].shape[0])

#         if verbose:
#             # Each (item_idx, t) maps to (t, t+1). If you selected 1000 items with T=50 in build_index_pairs,
#             # you should see 49,000 here.
#             print(f"[NS Dataset] index_pairs={len(self.index_pairs)} (t→t+1) pairs, T={self.T}, "
#                   f"normalize={self.normalize}, to_unit_range={self.to_unit_range}")

#     def __len__(self):
#         return len(self.index_pairs)

#     def _z_norm(self, arr):
#         return (arr - self.mean) / self.std

#     def _to_unit_range(self, arr):
#         # Optional squash to [-1, 1]
#         return np.tanh(arr)

#     def __getitem__(self, i):
#         item_idx, t = map(int, self.index_pairs[i])
#         seq = self.mem[item_idx]                 # (T, H, W)
#         y   = np.asarray(seq[t+1], dtype=np.float32)  # (H, W) target

#         if self.normalize:
#             y = self._z_norm(y)
#         if self.to_unit_range:
#             y = self._to_unit_range(y)

#         y = torch.from_numpy(y[None, ...])       # (1, H, W)  ← SINGLE CHANNEL
#         sample_id = torch.tensor(item_idx * self.T + t, dtype=torch.long)
#         return y, sample_id

class NavierStokesNextStepDDPM(Dataset):
    """
    Forecasting pairs: returns (x_t, y_{t+1}, sample_id).
    Keep your SparsityController for H/T splitting; this dataset does not build masks.

    Returns:
      - x_t: FloatTensor (1, H, W)  # previous frame
      - y_{t+1}: FloatTensor (1, H, W)  # next frame (target)
      - sample_id: int64 stable ID = item_idx*T + t
    """
    def __init__(self, memmap, index_pairs,
                 normalize=False, mean=0.0, std=1.0, to_unit_range=False,
                 max_samples=None, verbose=True):
        index_pairs = np.asarray(index_pairs, dtype=np.int64)
        if max_samples is not None:
            index_pairs = index_pairs[:max_samples]
        self.mem = memmap
        self.index_pairs = index_pairs
        self.normalize = bool(normalize)
        self.mean = float(mean)
        self.std = float(std)
        self.to_unit_range = bool(to_unit_range)

        # Assume fixed T for chosen items; keep for sample_id packing
        it0 = int(self.index_pairs[0, 0])
        self.T = int(self.mem[it0].shape[0])

        if verbose:
            # Each (item_idx, t) maps to (t, t+1)
            print(f"[NS Forecast] index_pairs={len(self.index_pairs)} (t→t+1) pairs, T={self.T}, "
                  f"normalize={self.normalize}, to_unit_range={self.to_unit_range}")

    def __len__(self):
        return len(self.index_pairs)

    def _z_norm(self, arr):
        return (arr - self.mean) / self.std

    def _to_unit_range(self, arr):
        # Optional squash to [-1, 1]
        return np.tanh(arr)

    def __getitem__(self, i):
        item_idx, t = map(int, self.index_pairs[i])
        seq = self.mem[item_idx]                 # (T, H, W)
        x = np.asarray(seq[t],     dtype=np.float32)  # (H, W) previous
        y = np.asarray(seq[t + 1], dtype=np.float32)  # (H, W) next (target)

        if self.normalize:
            x = self._z_norm(x); y = self._z_norm(y)
        if self.to_unit_range:
            x = self._to_unit_range(x); y = self._to_unit_range(y)

        x = torch.from_numpy(x[None, ...])       # (1, H, W)
        y = torch.from_numpy(y[None, ...])       # (1, H, W)
        sample_id = torch.tensor(item_idx * self.T + t, dtype=torch.long)
        return x, y, sample_id






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
            # no torchvision transform here; NS is already numeric (64x64)
            return NavierStokesNextStepDDPM(**kwargs)

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
