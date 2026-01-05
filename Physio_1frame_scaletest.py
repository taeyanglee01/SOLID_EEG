import lmdb
import datetime
import argparse
import pandas as pd
import numpy as np
import random
import re
from collections import defaultdict
from typing import List, Tuple, Union
import argparse

import scipy.io
import pickle
import os
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm

from torch.optim import AdamW
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import math
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

print("CLIP TRUE SETTING!")
print("z / 3 working NOT TANH")

#################### Argparse specific functions ####################

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser_args(parser):
    parser.add_argument('--seed', type=int, default=41, help='random seed(default: 41)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size(default: 16)')
    parser.add_argument('--lr', type=float, default=2e-4, help='old learning rate(default: 2e-4)')
    parser.add_argument('--max_lr', type=float, default=4e-4, help='max learning rate(default: 4e-4)')
    parser.add_argument('--min_lr', type=float, default=8e-6, help='min learning rate(default: 8e-6)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay(default: 1e-4)')
    parser.add_argument('--dataset_dir', type=str, default='./', help='lmdb dataset dir')
    parser.add_argument('--result_dir', type=str, default='./', help='result saving dir')
    parser.add_argument('--keep_ratio', type=float, default=0.9, help='keeping ratio for data')
    parser.add_argument('--squash_tanh', type=str2bool, default=True, help='tanh normalization')
    parser.add_argument('--data_scaling_factor', type=int, default=100, help='scaling factor for data')
    # step relevent arguments
    parser.add_argument('--time_steps', type=int, default=1000, help='diffusion time steps')
    parser.add_argument('--total_steps', type=int, default=100000, help='learning total steps')

    # logging relevent arguments
    parser.add_argument('--log_every', type=int, default=200, help='log steps stride')
    parser.add_argument('--eval_every', type=int, default=1000, help='eval steps stride')
    parser.add_argument('--save_samples_every', type=int, default=1000, help='save samples steps stride')
    parser.add_argument('--be_weight', type=int, default=0, help='be weight')

    return parser

    # TODO :
    # keep ratio, sparcity patterns and loss mode can be handled in argparse
    # use mae or crps also handled in this part
    # in channel for unet also

parser = argparse.ArgumentParser(description='SOLID for EEG')
parser = get_parser_args(parser)
params = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(params.result_dir, exist_ok=True)


#################### Util functions and constants ####################

TORCHEEG_2DGRID = [
    ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'],
    ['-', '-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-', '-'],
    ['-', '-', 'AF7', '-', 'AF3', 'AFZ', 'AF4', '-', 'AF8', '-', '-'],
    ['F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10'],
    ['FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10'], 
    ['T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10'],
    ['TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10'], 
    ['P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10'],
    ['-', '-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-', '-'],
    ['-', '-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-', '-'],
    ['-', '-', '-', '-', '-', 'IZ', '-', '-', '-', '-', '-']
    ]

def build_channel_to_rc(grid_2d):
    ch2rc = {}
    H = len(grid_2d)
    W = len(grid_2d[0])
    for r in range(H):
        for c in range(W):
            ch = grid_2d[r][c]
            if ch != '-' and ch is not None:
                ch2rc[str(ch).strip().upper()] = (r, c)
    return ch2rc, H, W

CHANNEL_TO_RC, H, W = build_channel_to_rc(TORCHEEG_2DGRID)

def to_tensor(array):
    return torch.from_numpy(array).float()

def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'set seed {seed} is done')

KeyT = Union[str, bytes, bytearray]

_KEY_RE = re.compile(
   r"^S(?P<sub_id>\d{3})R\d{2}-\d+$"
)

def _decode_key(k: KeyT) -> str:
    if isinstance(k, (bytes, bytearray)):
        return k.decode("utf-8", errors="ignore")
    return k

def _extract_sub_id(k: KeyT) -> str:
    s = _decode_key(k)
    m = _KEY_RE.match(s)
    if m is None:
        raise ValueError(f"Key does not match expected patterns: {s}")
    return m.group("sub_id")


def train_test_split_by_fold_num(
    fold_num: int,
    lmdb_keys: List[KeyT],
    maxFold: int,
    split_by_sub: bool = True,
    seed: int = 41
) -> Tuple[List[KeyT], List[KeyT]]:
    """
    True k-fold cross-validation split.

    Args:
        fold_num: test fold index (0 <= fold_num < maxFold)
        lmdb_keys: LMDB key list
        maxFold: total number of folds (k)
        split_by_sub: True → subject-wise k-fold, False → key-wise k-fold

    Returns:
        train_key_list, test_key_list
    """
    if maxFold < 2:
        raise ValueError("maxFold must be >= 2.")
    if fold_num < 0 or fold_num >= maxFold:
        raise ValueError(f"fold_num must be in [0, {maxFold-1}]")

    keys = list(lmdb_keys)

    rng = np.random.default_rng(seed)

    if split_by_sub:
        # -------- subject-wise k-fold --------
        sub_to_keys = defaultdict(list)
        invalid = []

        for k in keys:
            try:
                sid = _extract_sub_id(k)
                sub_to_keys[sid].append(k)
            except ValueError:
                invalid.append(_decode_key(k))

        if invalid:
            ex = "\n".join(invalid[:10])
            raise ValueError(
                f"Found {len(invalid)} invalid keys. Examples:\n{ex}"
            )

        subjects = np.array(list(sub_to_keys.keys()), dtype=object)
        rng.shuffle(subjects)

        subj_folds = np.array_split(subjects, maxFold)
        test_subjects = set(subj_folds[fold_num].tolist())

        train_keys, test_keys = [], []
        for sid, ks in sub_to_keys.items():
            (test_keys if sid in test_subjects else train_keys).extend(ks)

        return train_keys, test_keys

    else:
        # -------- key-wise k-fold --------
        idx = np.arange(len(keys))
        rng.shuffle(idx)

        folds = np.array_split(idx, maxFold)
        test_idx = set(folds[fold_num].tolist())

        train_keys = [keys[i] for i in idx if i not in test_idx]
        test_keys  = [keys[i] for i in idx if i in test_idx]

        return train_keys, test_keys

#################### Dataset classes ####################

print("DATASET SETTING RUNNING")

class Physio_1sec_raw_for_SOLID_from_lmdb(Dataset):
    def __init__(
            self,
            lmdb_dir: str,
            maxfold: int,
            targetfold: int,
            seed: int,
            train: bool,
            split_by_sub: bool,
            seg_len_pts: int = 1,
            stride_pts: int = 1,
            mean: float = 0.0,
            std: float = 1.0
    ):
        random_seed(seed)
        self.seed = seed
        self.lmdb_dir = lmdb_dir
        self.db = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.lmdb_keys = pickle.loads(txn.get('__keys__'.encode()))

        self.train = train
        self.split_by_sub = split_by_sub

        self.maxfold = maxfold
        self.targetfold = targetfold

        self.seg_len_pts = seg_len_pts
        self.stride_pts = stride_pts

        self.data, self.data_meta = self.make_segments_by_fold(
            self.targetfold, self.lmdb_keys, self.maxfold, self.split_by_sub, self.seed
        )

        if self.train:
            self._compute_mean_std()
        else:
            self.mean = float(mean)
            self.std  = float(std)

    def make_segments_by_fold(self, fold, lmdb_keys, maxfold, split_by_sub, seed):
        train_keys, test_keys = train_test_split_by_fold_num(
            fold, lmdb_keys, maxfold, split_by_sub, seed
        )

        use_keys = train_keys if self.train else test_keys

        all_segs = []
        all_meta = []

        for key in use_keys:
            seg_list, meta_list = self.segment_by_points_from_key(
                    key, self.db,
                    seg_len_pts=self.seg_len_pts,
                    stride_pts=self.stride_pts
                )
            all_segs += seg_list
            all_meta += meta_list

        return all_segs, all_meta

    def lmdb_get(self, env, key):
        if isinstance(key, str):
            key = key.encode("utf-8")
        with env.begin(write=False) as txn:
            v = txn.get(key)
        if v is None:
            raise KeyError(f"Key not found: {key}")
        return pickle.loads(v)

    def segment_by_points_from_key(self, key, lmdb_db, seg_len_pts=1, stride_pts=1):
        """
        LMDB sample['sample'] shape: (C, T, Fs)
        -> flatten to (C, T*Fs)
        -> segment by points:
        x = flat[:, p:p+seg_len_pts]  # (C, seg_len_pts)
        default seg_len_pts=1 => (C, 1) per sample
        """
        sample_for_key = self.lmdb_get(lmdb_db, key)

        channel_name = sample_for_key['data_info']['channel_names']  # len=C
        eeg = sample_for_key['sample']  # (C, T, Fs), numpy or torch

        if isinstance(eeg, torch.Tensor):
            eeg_t = eeg.float()
        else:
            eeg_t = torch.from_numpy(eeg).to(torch.float32)

        C, T, Fs = eeg_t.shape

        # (C, T*Fs)
        flat = eeg_t.reshape(C, T * Fs)

        L = flat.shape[1]
        if L < seg_len_pts:
            return [], []

        seg_list = []
        meta_list = []

        for p in range(0, L - seg_len_pts + 1, stride_pts):
            x = flat[:, p:p + seg_len_pts]   # (C, seg_len_pts)
            seg_list.append(x)
            meta_list.append(channel_name)   # 필요하면 여기 더 풍부한 meta로 바꿔도 됨

        return seg_list, meta_list

    def _compute_mean_std(self):
        """
        Compute mean/std over the entire TRAIN dataset.
        Uses streaming to avoid memory blow-up.
        """
        print("[PhysioDataset] Computing mean/std from TRAIN set...")

        total_sum = 0.0
        total_sq  = 0.0
        total_n   = 0

        for x in self.data:
            # x: (C, L)
            x = x.float()

            total_sum += x.sum().item()
            total_sq  += (x ** 2).sum().item()
            total_n   += x.numel()

        mean = total_sum / total_n
        var  = total_sq / total_n - mean ** 2
        std  = (var ** 0.5) if var > 0 else 1.0

        self.mean = float(mean)
        self.std  = float(std)

        print(f"[Train Dataset] mean={self.mean:.6f}, std={self.std:.6f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x  = self.data[idx]        # (C, 200) if seg_len_sec=1 and Fs=200
        xm = self.data_meta[idx]   # channel names
        return x, xm

def splat_eeg_grid(eeg_cl, channel_names, channel_to_rc=CHANNEL_TO_RC, H=H, W=W):
    """
    eeg_cl: (C, L) torch.Tensor (권장)
    channel_names: list[str] len=C
    returns:
      grid: (L, H, W)
      mask: (H, W)
    """
    if not torch.is_tensor(eeg_cl):
        eeg_cl = torch.as_tensor(eeg_cl)

    assert eeg_cl.dim() == 2, f"Expected (C,L), got {tuple(eeg_cl.shape)}"
    C, L = eeg_cl.shape
    device = eeg_cl.device

    grid = torch.zeros((L, H, W), dtype=eeg_cl.dtype, device=device)
    cnt  = torch.zeros((H, W), dtype=torch.float32, device=device)

    for ci in range(C):
        ch = str(channel_names[ci]).strip().upper()
        if ch in channel_to_rc:
            r, c = channel_to_rc[ch]
            grid[:, r, c] += eeg_cl[ci, :]
            cnt[r, c] += 1.0

    mask = (cnt > 0).float()
    grid = torch.where(cnt > 0, grid / torch.clamp(cnt, min=1.0), grid)
    return grid, mask

class EEGToGridCtx9_1sec(Dataset):
    """
    NEW VERSION for 1-sec raw base dataset.

    base[idx] -> (x, xm)
      x  : (C, L)  (torch or numpy)  e.g. (64, 200)
      xm : channel list (len=C)

    return:
      x0        : (L, H, W)   full target grid in tanh space (optional)
      loss_mask : (1, H, W)   1=supervise bins (typically UNOBSERVED electrode bins)
      cond      : (L+3, H, W) = [lat_map(1), lon_map(1), obs_grid(L), obs_mask(1)]
      mean, std
    """
    def __init__(
        self,
        base_dataset,
        squash_tanh: bool = True,
        channel_to_rc=CHANNEL_TO_RC,
        keep_ratio: float = 0.9,          # fraction of electrode bins observed
        seed: int = 0,
        loss_mode: str = 'all' # True면 (1-obs)에서만 loss
    ):
        self.base = base_dataset
        self.squash = squash_tanh
        self.channel_to_rc = channel_to_rc
        self.keep_ratio = keep_ratio
        self.seed = seed
        self.loss_mode = loss_mode

        self.mean = float(getattr(self.base, "mean", 0.0))
        self.std  = float(getattr(self.base, "std",  1.0))

        lat = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
        lon = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
        self.lat_map = lat
        self.lon_map = lon

    def __len__(self):
        return len(self.base)

    def _sample_obs_mask(self, tgt_mask_hw: torch.Tensor, idx: int) -> torch.Tensor:
        """
        tgt_mask_hw: (H,W)  1 where electrode exists
        returns obs_mask_hw: (H,W) subset of tgt_mask_hw set to 1
        """
        g = torch.Generator(device=tgt_mask_hw.device)
        g.manual_seed(self.seed + idx)

        valid = torch.nonzero(tgt_mask_hw > 0.5, as_tuple=False)  # (N,2)
        N = valid.shape[0]
        if N == 0:
            return torch.zeros_like(tgt_mask_hw)

        k = max(1, int(round(self.keep_ratio * N)))
        perm = torch.randperm(N, generator=g, device=tgt_mask_hw.device)
        chosen = valid[perm[:k]]

        obs = torch.zeros_like(tgt_mask_hw)
        obs[chosen[:, 0], chosen[:, 1]] = 1.0
        return obs

    def __getitem__(self, idx):
        # base: (C,L), channel_names
        x, xm = self.base[idx]

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.to(torch.float32)

        # ---- full target grid/mask ----
        # grid: (L,H,W), mask: (H,W)
        full_grid, tgt_mask_hw = splat_eeg_grid(
            x, xm, channel_to_rc=self.channel_to_rc, H=H, W=W
        )

        # (optional) squash to tanh space
        if self.squash:
            z = (full_grid - self.mean) / (self.std + 1e-6)
            x0 = torch.clamp(z / 3, -1.0, 1.0) # divide 3 case
            # x0 = torch.tanh(z) # tanh case
        else:
            x0 = full_grid/params.data_scaling_factor  # (L,H,W)

        # DEBUG PRINT
        if self.squash and idx == 0:
            with torch.no_grad():
                z = (full_grid - self.mean) / (self.std + 1e-6)
                sat = (x0.abs() > 0.98).float().mean().item()
                print(f"[DS] mean={self.mean:.4f} std={self.std:.4f} | "
                    f"z: min={z.min():.2f} max={z.max():.2f} p(|z|>3)={(z.abs()>3).float().mean():.3f} | "
                    f"x0: min={x0.min():.2f} max={x0.max():.2f} sat(|x0|>0.98)={sat:.3f}")

        # ---- observed subset mask (spatial only) ----
        obs_mask_hw = self._sample_obs_mask(tgt_mask_hw, idx)     # (H,W)

        # ---- observed input grid ----
        obs_grid = x0 * obs_mask_hw.unsqueeze(0)                  # (L,H,W)

        L = full_grid.shape[0]
        t_index = torch.linspace(
            0.0, 1.0, steps=L, device=full_grid.device
        ).view(L, 1, 1).expand(L, H, W)

        # ---- cond ----
        
        cond = torch.cat([
            # self.lat_map.unsqueeze(0),            # (1,H,W)
             #self.lon_map.unsqueeze(0),            # (1,H,W)
            # t_index,                              # (L,H,W)
            obs_grid,                             # (L,H,W) # FIXME : 시간축 이거 필요 없을지도? 뺴고 한 번 넣고 한 번 해보자
            # obs_mask_hw.unsqueeze(0),             # (1,H,W)
        ], dim=0)                                 # (L+3,H,W)
        
        '''
        cond = torch.cat([
            self.lat_map.unsqueeze(0),            # (1,H,W)
            self.lon_map.unsqueeze(0),            # (1,H,W)
            obs_mask_hw.unsqueeze(0),             # (1,H,W)
        ], dim=0)  
        '''

        # ---- loss mask: where to supervise ----
        # electrode bins only (tgt_mask_hw) + (unobserved OR observed)
        if self.loss_mode == 'all':
            loss_mask_hw = tgt_mask_hw      # (H,W)
        elif self.loss_mode == 'obs':
            loss_mask_hw = obs_mask_hw * tgt_mask_hw            # (H,W)
        elif self.loss_mode == "unobs":
            loss_mask_hw = (1.0 - obs_mask_hw) * tgt_mask_hw              # (H,W)
        else :
            raise ValueError('loss_mode is weird')

        return x0, loss_mask_hw.unsqueeze(0), cond, self.mean, self.std

sample_train_dataset_1sec = Physio_1sec_raw_for_SOLID_from_lmdb(lmdb_dir=params.dataset_dir,
                                                     maxfold=10,
                                                     targetfold=0,
                                                     seed=params.seed,
                                                     train=True,
                                                     split_by_sub=True)

sample_test_dataset_1sec = Physio_1sec_raw_for_SOLID_from_lmdb(lmdb_dir=params.dataset_dir,
                                                     maxfold=10,
                                                     targetfold=0,
                                                     seed=params.seed,
                                                     train=False,
                                                     split_by_sub=True,
                                                     mean=sample_train_dataset_1sec.mean,
                                                     std=sample_train_dataset_1sec.std)

train_grid_dataset = EEGToGridCtx9_1sec(base_dataset=sample_train_dataset_1sec, 
                                        squash_tanh=params.squash_tanh, # TODO : tanh is bset option??
                                        channel_to_rc=CHANNEL_TO_RC,
                                        keep_ratio=params.keep_ratio, # TODO : sparsity option should be added
                                        seed=params.seed,
                                        loss_mode='all')

test_grid_dataset = EEGToGridCtx9_1sec(base_dataset=sample_test_dataset_1sec, 
                                        squash_tanh=params.squash_tanh,
                                        channel_to_rc=CHANNEL_TO_RC,
                                        keep_ratio=params.keep_ratio,
                                        seed=params.seed,
                                        loss_mode='all')

train_loader = DataLoader(train_grid_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_grid_dataset, batch_size=params.batch_size*2, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)


print(len(train_grid_dataset))
print(len(test_grid_dataset))

#################### Model implementation from Kevin ####################

print("MODEL SETTING RUNNING")

# ============================================================
# 3) UNet (no attention; rectangular-friendly)
# ============================================================
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=0.0, groups=32):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim else None
        self.norm1 = nn.GroupNorm(groups, dim);     self.conv1 = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, dim_out); self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.act = nn.SiLU()
        self.res = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, t_emb=None):
        h = self.conv1(self.act(self.norm1(x)))
        if self.mlp is not None and t_emb is not None:
            h = h + self.mlp(t_emb)[..., None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.res(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, base_dim):
        super().__init__()
        self.out_dim = base_dim
    def forward(self, t):  # t: (B,)
        # classic transformer-style PE on scalar t
        half = self.out_dim // 2
        device = t.device
        freqs = torch.exp(torch.arange(half, device=device).float()
                          * -(math.log(10000.0) / max(1, half-1)))
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (B, 2*half)
        if emb.shape[1] < self.out_dim:
            emb = F.pad(emb, (0, self.out_dim - emb.shape[1]))
        return emb

class TimeMLP(nn.Module):
    def __init__(self, base_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, base_dim*4), nn.SiLU(),
            nn.Linear(base_dim*4, base_dim*4)
        )
    def forward(self, t):  # (B,)
        return self.net(t[:,None].float())

class UNet(nn.Module):
    def __init__(self, base_dim=128, dim_mults=(1,2,4),
                 in_channels=1+20, image_size=(H,W), dropout=0.0, groups=32):
        super().__init__()
        self.image_h, self.image_w = image_size
        self.time_dim = base_dim * 4

        # self.time_pe  = SinusoidalTimeEmbedding(base_dim)
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(base_dim, self.time_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.time_dim, self.time_dim)
        # )
        self.time_mlp = TimeMLP(base_dim)
        self.init = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        self.downs = nn.ModuleList()
        in_ch = base_dim
        skip_channels = []
        for li, m in enumerate(dim_mults):
            out_ch = base_dim * m
            rb1 = ResnetBlock(in_ch, out_ch, self.time_dim, dropout, groups); self.downs.append(rb1); in_ch = out_ch; skip_channels.append(in_ch)
            rb2 = ResnetBlock(in_ch, out_ch, self.time_dim, dropout, groups); self.downs.append(rb2); in_ch = out_ch; skip_channels.append(in_ch)
            if li != len(dim_mults) - 1:
                self.downs.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))

        self.mid1 = ResnetBlock(in_ch, in_ch, self.time_dim, dropout, groups)
        self.mid2 = ResnetBlock(in_ch, in_ch, self.time_dim, dropout, groups)

        self.ups, self.kinds = nn.ModuleList(), []
        sc = skip_channels.copy()
        for li, m in enumerate(reversed(dim_mults)):
            out_ch = base_dim * m
            for _ in range(2):
                skip_ch = sc.pop()
                self.ups.append(ResnetBlock(in_ch + skip_ch, out_ch, self.time_dim, dropout, groups)); self.kinds.append('res')
                in_ch = out_ch
            if li != len(dim_mults) - 1:
                self.ups.append(nn.Upsample(scale_factor=2, mode='nearest')); self.kinds.append('up')
                self.ups.append(nn.Conv2d(in_ch, in_ch, 3, padding=1));       self.kinds.append('conv')

        self.final = nn.Sequential(nn.GroupNorm(groups, in_ch), nn.SiLU(), nn.Conv2d(in_ch, 1, 3, padding=1)) # output dim to 200 from 1

    def forward(self, x_cat, t):
        # t_emb = self.time_mlp(self.time_pe(t))
        t_emb = self.time_mlp(t)
        skips, h = [], self.init(x_cat)
        for layer in self.downs:
            if isinstance(layer, ResnetBlock):
                h = layer(h, t_emb); skips.append(h)
            else:
                h = layer(h)
        h = self.mid1(h, t_emb); h = self.mid2(h, t_emb)
        for kind, layer in zip(self.kinds, self.ups):
            if kind == 'res':
                s = skips.pop()
                if s.shape[-2:] != h.shape[-2:]:
                    s = F.interpolate(s, size=h.shape[-2:], mode='nearest')
                h = layer(torch.cat([h, s], dim=1), t_emb)
            elif kind == 'up':
                h = layer(h)
            else:
                h = layer(h)
        if h.shape[-2:] != (self.image_h, self.image_w):
            h = F.interpolate(h, size=(self.image_h, self.image_w), mode='nearest')
        return self.final(h)

# ============================================================
# 4) Diffusion core — noise only target channel; cond is clean
# ============================================================
class GaussianDiffusion(nn.Module):
    def __init__(self, unet, image_size=(H,W), time_steps=params.time_steps, loss_type='l2'):
        super().__init__()
        self.unet = unet
        self.H, self.W = image_size
        self.T = time_steps
        self.loss_type = loss_type

        beta  = self.linear_beta_schedule(time_steps)
        alpha = 1. - beta
        abar  = torch.cumprod(alpha, dim=0)
        abar_prev = F.pad(abar[:-1], (1,0), value=1.)

        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', abar)
        self.register_buffer('alpha_bar_prev', abar_prev)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(abar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1 - abar))
        self.register_buffer('sqrt_recip_alpha_bar', torch.sqrt(1. / abar))
        self.register_buffer('sqrt_recip_alpha_bar_min_1', torch.sqrt(1. / abar - 1))
        self.register_buffer('sqrt_recip_alpha', torch.sqrt(1. / alpha))
        self.register_buffer('beta_over_sqrt_one_minus_alpha_bar', beta / torch.sqrt(1. - abar))

    def linear_beta_schedule(self, T):
        scale = 1000 / T
        return torch.linspace(scale*1e-4, scale*2e-2, T, dtype=torch.float32)

    def q_sample(self, x0, t, noise):
        return self.sqrt_alpha_bar[t][:,None,None,None] * x0 + \
               self.sqrt_one_minus_alpha_bar[t][:,None,None,None] * noise

    def forward(self, x0, mask, cond):
        """
        x0:   (B,T,H,W) in tanh(z) space
        mask: (B,1,H,W)  (1=observed bin in target; 0=unobserved) # T -> 1
        cond: (B,3,H,W) = [lat, lon, obs_mask] # cond 자체를 제거해서 돌려보기 당장 interpolation하는 것은 아니니 지금은 제거해도 괜찮을 것
        """
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device).long()

        noise = torch.randn_like(x0)
        x_t   = self.q_sample(x0, t, noise)
        x_cat = torch.cat([x_t, cond], dim=1)  # noised target + clean cond
        # x_cat = x_t # cond 없이 실험

        pred = self.unet(x_cat, t)  # predict noise on target channel

        # DEBUG PRINT
        if torch.rand(()) < 0.001:
            with torch.no_grad():
                # x0는 tanh(z) 공간, noise는 N(0,1), pred도 ideally N(0,1) 근처
                print(f"[Diff] x0: ({x0.min().item():.2f},{x0.max().item():.2f}) "
                    f"sat={(x0.abs()>0.98).float().mean().item():.3f} | "
                    f"x_t: mean={x_t.mean().item():.3f} std={x_t.std().item():.3f} | "
                    f"noise: mean={noise.mean().item():.3f} std={noise.std().item():.3f} | "
                    f"pred:  mean={pred.mean().item():.3f} std={pred.std().item():.3f} "
                    f"pred_abs>5={(pred.abs()>5).float().mean().item():.3f}")

        if self.loss_type == 'l1':
            raw = F.l1_loss(noise, pred, reduction='none')
        elif self.loss_type == 'l2':
            raw = F.mse_loss(noise, pred, reduction='none')
        else:
            raw = F.smooth_l1_loss(noise, pred, reduction='none')

        w = mask + params.be_weight  # supervise observed bins + tiny everywhere
        return (raw * w).sum() / (w.sum() + 1e-8)

    @torch.inference_mode()
    def p_sample(self, xt, cond, t, clip=True):
        bt = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        x_cat = torch.cat([xt, cond], dim=1)
        # x_cat = xt
        pred_noise = self.unet(x_cat, bt)

        def bcast(x): return x.view(-1,1,1,1)
        if clip:
            x0 = bcast(self.sqrt_recip_alpha_bar[bt]) * xt - bcast(self.sqrt_recip_alpha_bar_min_1[bt]) * pred_noise
            x0 = x0.clamp(-1., 1.)
            c1 = self.beta[bt] * torch.sqrt(self.alpha_bar_prev[bt]) / (1. - self.alpha_bar[bt])
            c2 = torch.sqrt(self.alpha[bt]) * (1. - self.alpha_bar_prev[bt]) / (1. - self.alpha_bar[bt])
            # c2 = torch.sqrt(self.alpha[bt]) * (1. - self.alpha_bar_prev[bt]) / (1. - self.alpha[bt]) # FIXME : original code
            mean = bcast(c1) * x0 + bcast(c2) * xt
        else:
            mean = bcast(self.sqrt_recip_alpha[bt]) * (xt - bcast(self.beta_over_sqrt_one_minus_alpha_bar[bt]) * pred_noise)
        var = self.beta[bt] * ((1. - self.alpha_bar_prev[bt]) / (1. - self.alpha_bar[bt]))
        noise = torch.randn_like(xt) if t > 0 else 0.
        return mean + torch.sqrt(bcast(var)) * noise

    @torch.inference_mode()
    def sample(self, cond, clip=False, debug=False):  # clip=False often gives crisper fields
        b = cond.size(0)
        x = torch.randn((b,1,self.H,self.W), device=cond.device)
        watch = set([self.T-1, self.T//2, 0])
        for t in reversed(range(self.T)):
            x = self.p_sample(x, cond, t, clip=clip)
            if debug and (t in watch):
                sat = (x.abs() > 0.98).float().mean().item()
                print(f"[SAMPLE] t={t:4d} x: min={x.min().item():.3f} max={x.max().item():.3f} "
                    f"sat(|x|>0.98)={sat:.3f}")

        x = x.clamp(-1, 1)
        if debug:
            sat = (x.abs() > 0.98).float().mean().item()
            print(f"[SAMPLE] final clamp x: sat(|x|>0.98)={sat:.3f}")
        return x
        for t in reversed(range(self.T)):
            x = self.p_sample(x, cond, t, clip=clip)
        return x.clamp(-1, 1)

# ============================================================
# 6) Build model + diffusion + optimizer
# ============================================================
print("TRANING SETTING RUNNING")

IN_CHANNELS = 1 + 1   # target(noised) + lat/lon + 9 past grids + 9 past masks = 21
unet = UNet(base_dim=128, dim_mults=(1,2,4), in_channels=IN_CHANNELS, image_size=(H,W)).to(DEVICE)
diffusion = GaussianDiffusion(unet, image_size=(H,W), time_steps=params.time_steps, loss_type='l2').to(DEVICE)

# ---- CosineAnnealingWarmupRestarts setup ----

warmup_steps = max(1, int(0.1 * params.total_steps))

opt = AdamW(diffusion.parameters(), lr=params.max_lr, betas=(0.9, 0.999), weight_decay=params.wd)

sched = CosineAnnealingWarmupRestarts(
    optimizer=opt,
    first_cycle_steps=params.total_steps,  # single full-length cosine cycle
    max_lr=params.max_lr,
    min_lr=params.min_lr,
    warmup_steps=warmup_steps,
    gamma=1.0                       # no decay across cycles since we use one cycle
)

# ============================================================
# 7) Utilities: invert tanh->raw and metrics
# ============================================================
def inv_tanh_to_raw(x_tanh, mean, std, debug=False, tag="INV"):
    '''
    if debug:
        with torch.no_grad():
            sat = (x_tanh.abs() > 0.98).float().mean().item()
            mx  = x_tanh.abs().max().item()
            print(f"[{tag}] x_tanh: min={x_tanh.min().item():.3f} max={x_tanh.max().item():.3f} "
                  f"sat(|x|>0.98)={sat:.3f} max|x|={mx:.6f}")
# TODO : z norm 이후에 min max까지 맞추는 것은 굳이이므로 뺴서 실험해보기 -> input의 통계량을 시각화해서 보여드리기
    x_clamped = x_tanh.clamp(-0.999, 0.999)
    z = torch.atanh(x_clamped)

    # DEBUG PRINT
    if debug:
        with torch.no_grad():
            # atanh 결과가 크면 raw가 터질 수 있음
            print(f"[{tag}] z(atanh): min={z.min().item():.3f} max={z.max().item():.3f} "
                  f"mean={z.mean().item():.3f} std={z.std().item():.3f}")
    raw = z * std + mean
    if debug:
        with torch.no_grad():
            print(f"[{tag}] raw: min={raw.min().item():.3f} max={raw.max().item():.3f} "
                  f"mean={raw.mean().item():.3f} std={raw.std().item():.3f}")
    return z * std + mean
    '''
    if debug:
        with torch.no_grad():
            sat = (x_tanh.abs() >= 0.999).float().mean().item()
            mx  = x_tanh.abs().max().item()
            print(f"[{tag}] x_clamp: min={x_tanh.min().item():.3f} "
                  f"max={x_tanh.max().item():.3f} "
                  f"sat(|x|>=0.999)={sat:.3f} max|x|={mx:.6f}")

    # clamp inverse (linear region only)
    z = 3.0 * x_tanh

    if debug:
        with torch.no_grad():
            print(f"[{tag}] z(pseudo-inv): min={z.min().item():.3f} "
                  f"max={z.max().item():.3f} "
                  f"mean={z.mean().item():.3f} std={z.std().item():.3f}")

    raw = z * std + mean

    if debug:
        with torch.no_grad():
            print(f"[{tag}] raw: min={raw.min().item():.3f} "
                  f"max={raw.max().item():.3f} "
                  f"mean={raw.mean().item():.3f} std={raw.std().item():.3f}")

    return raw

@torch.no_grad()
def eval_rmse(diffusion, loader): # add mini batch to check fast
    mse_sum_raw = 0.0; w_sum = 0.0
    mse_sum_norm = 0.0
    for x0_te, m_te, c_te, mu_te, std_te in loader:
        x0_te  = x0_te.to(DEVICE)   # tanh(z)
        m_te   = m_te.to(DEVICE)
        c_te   = c_te.to(DEVICE)
        mu_te  = mu_te.to(DEVICE)[:,None,None,None]
        std_te = std_te.to(DEVICE)[:,None,None,None]

        xhat = diffusion.sample(c_te, clip=True)             # tanh(z)
        # raw µV
        if params.squash_tanh:
            raw_hat = inv_tanh_to_raw(xhat,  mu_te, std_te)
            raw_gt  = inv_tanh_to_raw(x0_te, mu_te, std_te)
        else:
            raw_hat = xhat*params.data_scaling_factor
            raw_gt = x0_te*params.data_scaling_factor

        mse_sum_raw  += ((raw_hat - raw_gt)**2 * m_te).sum().item()
        w_sum        += m_te.sum().item()
        # normalized (z) space RMSE (mask)
        zhat = torch.atanh(xhat.clamp(-0.999, 0.999))
        zgt  = torch.atanh(x0_te.clamp(-0.999, 0.999))
        mse_sum_norm += ((zhat - zgt)**2 * m_te).sum().item()

    rmse_raw  = math.sqrt(mse_sum_raw / max(w_sum, 1e-8))
    rmse_norm = math.sqrt(mse_sum_norm / max(w_sum, 1e-8))
    return rmse_raw, rmse_norm, int(w_sum)

@torch.no_grad()
def eval_rmse_with_pbar(diffusion, loader, max_batches=None, show_pbar=True, pbar_position=1):
    mse_sum_raw = 0.0; w_sum = 0.0
    mse_sum_norm = 0.0

    it = loader
    if show_pbar:
        total = len(loader) if hasattr(loader, "__len__") else None
        it = tqdm(loader, total=total, desc="eval", leave=False, position=pbar_position)

    with torch.inference_mode():
        for bi, (x0_te, m_te, c_te, mu_te, std_te) in enumerate(it):
            if (max_batches is not None) and (bi >= max_batches):
                break

            x0_te  = x0_te.to(DEVICE)
            m_te   = m_te.to(DEVICE)
            c_te   = c_te.to(DEVICE)

            mu_te  = mu_te.to(DEVICE)[:, None, None, None]
            std_te = std_te.to(DEVICE)[:, None, None, None]

            xhat = diffusion.sample(c_te, clip=True, debug=(bi==0))

            if bi == 0:
                with torch.no_grad():
                    sat = (xhat.abs() > 0.98).float().mean().item()
                    print(f"[EVAL] xhat(tanh): min={xhat.min().item():.3f} max={xhat.max().item():.3f} "
                        f"sat(|x|>0.98)={sat:.3f} | clip=True")

                    # mask 적용이 제대로 전극 위치에만 되는지 확인
                    print(f"[EVAL] mask: mean={m_te.mean().item():.3f} sum={m_te.sum().item():.0f} "
                        f"shape={tuple(m_te.shape)}")

            if params.squash_tanh:
                dbg = (bi == 0)
                raw_hat = inv_tanh_to_raw(xhat,  mu_te, std_te, debug=dbg, tag="EVAL_hat")
                raw_gt  = inv_tanh_to_raw(x0_te, mu_te, std_te, debug=dbg, tag="EVAL_gt")
                # raw_hat = inv_tanh_to_raw(xhat,  mu_te, std_te)
                # raw_gt  = inv_tanh_to_raw(x0_te, mu_te, std_te)
            else:
                raw_hat = xhat*params.data_scaling_factor
                raw_gt = x0_te*params.data_scaling_factor

            mse_sum_raw  += ((raw_hat - raw_gt)**2 * m_te).sum().item()
            w_sum        += m_te.sum().item()

            zhat = torch.atanh(xhat.clamp(-0.999, 0.999))
            zgt  = torch.atanh(x0_te.clamp(-0.999, 0.999))
            mse_sum_norm += ((zhat - zgt)**2 * m_te).sum().item()

            if show_pbar:
                it.set_postfix(w=int(w_sum), rmse=math.sqrt(mse_sum_raw / max(w_sum, 1e-8)))

    rmse_raw  = math.sqrt(mse_sum_raw / max(w_sum, 1e-8))
    rmse_norm = math.sqrt(mse_sum_norm / max(w_sum, 1e-8))
    diffusion.train()
    return rmse_raw, rmse_norm, int(w_sum)

@torch.no_grad()
def eval_rmse_minibatch(diffusion, loader, max_batches=None): # add mini batch to check fast
    mse_sum_raw = 0.0; w_sum = 0.0
    mse_sum_norm = 0.0
    for i, batch in enumerate(loader):
        if (max_batches is not None) and (i >= max_batches):
            break
        x0_te, m_te, c_te, mu_te, std_te = batch
    # for x0_te, m_te, c_te, mu_te, std_te in loader:
        x0_te  = x0_te.to(DEVICE)   # tanh(z)
        m_te   = m_te.to(DEVICE)
        c_te   = c_te.to(DEVICE)
        mu_te  = mu_te.to(DEVICE)[:,None,None,None]
        std_te = std_te.to(DEVICE)[:,None,None,None]

        xhat = diffusion.sample(c_te, clip=True)             # tanh(z)
        # raw µg/m³
        if params.squash_tanh:
            raw_hat = inv_tanh_to_raw(xhat,  mu_te, std_te)
            raw_gt  = inv_tanh_to_raw(x0_te, mu_te, std_te)
        else:
            raw_hat = xhat*params.data_scaling_factor
            raw_gt = x0_te*params.data_scaling_factor

        mse_sum_raw  += ((raw_hat - raw_gt)**2 * m_te).sum().item()
        w_sum        += m_te.sum().item()
        # normalized (z) space RMSE (mask)
        zhat = torch.atanh(xhat.clamp(-0.999, 0.999))
        zgt  = torch.atanh(x0_te.clamp(-0.999, 0.999))
        mse_sum_norm += ((zhat - zgt)**2 * m_te).sum().item()

    rmse_raw  = math.sqrt(mse_sum_raw / max(w_sum, 1e-8))
    rmse_norm = math.sqrt(mse_sum_norm / max(w_sum, 1e-8))
    return rmse_raw, rmse_norm, int(w_sum)

import math

@torch.no_grad()
def _crps_from_ensemble(y_flat, samples_flat):
    """
    y_flat:        (N,) ground-truth vector (masked entries only later)
    samples_flat:  (K,N) ensemble samples
    returns:       (N,) CRPS per entry
    """
    K = samples_flat.shape[0]
    # term1 = E|X - y|  ≈ (1/K) Σ_i |x_i - y|
    term1 = (samples_flat - y_flat.unsqueeze(0)).abs().mean(dim=0)  # (N,)
    # term2 = 0.5 * E|X - X'| ≈ 0.5 * (1/K^2) Σ_ij |x_i - x_j|
    diffs = samples_flat.unsqueeze(0) - samples_flat.unsqueeze(1)   # (K,K,N)
    term2 = 0.5 * diffs.abs().mean(dim=(0,1))                      # (N,)
    return term1 - term2                                            # (N,)

@torch.no_grad()
def eval_crps_and_points(diffusion, loader, K=10, clip=False):
    """
    Returns masked dataset-averaged:
      CRPS_raw, CRPS_norm, MAE_raw, RMSE_raw, MAE_norm, RMSE_norm, n_obs_bins
    """
    crps_raw_sum = 0.0
    crps_norm_sum = 0.0
    mae_raw_sum = 0.0
    rmse_raw_sum = 0.0
    mae_norm_sum = 0.0
    rmse_norm_sum = 0.0
    w_sum = 0.0

    for x0_te, m_te, c_te, mu_te, std_te in loader:
        x0_te  = x0_te.to(DEVICE)          # (B,1,H,W), tanh(z)
        m_te   = m_te.to(DEVICE)           # (B,1,H,W) mask
        c_te   = c_te.to(DEVICE)           # (B,20,H,W)
        mu_te  = mu_te.to(DEVICE)[:,None,None,None]
        std_te = std_te.to(DEVICE)[:,None,None,None]

        B, _, H, W = x0_te.shape
        N = B*H*W
        mask_flat = m_te.view(N).bool()

        # K samples
        samples = []
        for _ in range(K):
            xhat = diffusion.sample(c_te, clip=clip)             # (B,1,H,W) tanh(z)
            samples.append(xhat)
        S = torch.stack(samples, dim=0)                          # (K,B,1,H,W)

        # normalized (z)
        z_gt  = torch.atanh(x0_te.clamp(-0.999, 0.999))          # (B,1,H,W)
        z_smp = torch.atanh(S.clamp(-0.999, 0.999))              # (K,B,1,H,W)

        # raw μg/m³
        raw_gt  = z_gt * std_te + mu_te                          # (B,1,H,W)
        raw_smp = z_smp * std_te + mu_te                         # (K,B,1,H,W)

        # flatten
        z_gt_f   = z_gt.view(N)
        raw_gt_f = raw_gt.view(N)
        z_smp_f  = z_smp.view(K, N)
        raw_smp_f= raw_smp.view(K, N)

        # CRPS (masked mean)
        crps_norm = _crps_from_ensemble(z_gt_f,   z_smp_f)[mask_flat].mean()
        crps_raw  = _crps_from_ensemble(raw_gt_f, raw_smp_f)[mask_flat].mean()
        crps_norm_sum += crps_norm.item() * mask_flat.sum().item()
        crps_raw_sum  += crps_raw.item()  * mask_flat.sum().item()

        # point forecast = ensemble mean
        z_mean   = z_smp.mean(dim=0).view(N)
        raw_mean = raw_smp.mean(dim=0).view(N)

        # MAE/RMSE (masked)
        mae_norm_sum  += (z_mean - z_gt_f).abs()[mask_flat].sum().item()
        rmse_norm_sum += ((z_mean - z_gt_f)**2)[mask_flat].sum().item()
        mae_raw_sum   += (raw_mean - raw_gt_f).abs()[mask_flat].sum().item()
        rmse_raw_sum  += ((raw_mean - raw_gt_f)**2)[mask_flat].sum().item()

        w_sum += mask_flat.sum().item()

    CRPS_raw  = crps_raw_sum  / max(w_sum, 1e-8)
    CRPS_norm = crps_norm_sum / max(w_sum, 1e-8)
    MAE_raw   = mae_raw_sum   / max(w_sum, 1e-8)
    MAE_norm  = mae_norm_sum  / max(w_sum, 1e-8)
    RMSE_raw  = math.sqrt(rmse_raw_sum  / max(w_sum, 1e-8))
    RMSE_norm = math.sqrt(rmse_norm_sum / max(w_sum, 1e-8))

    return dict(
        CRPS_raw=CRPS_raw, CRPS_norm=CRPS_norm,
        MAE_raw=MAE_raw, RMSE_raw=RMSE_raw,
        MAE_norm=MAE_norm, RMSE_norm=RMSE_norm,
        K=K, n_obs_bins=int(w_sum),
    )



# ============================================================
# 8) Training loop with periodic eval + checkpoint
# ============================================================
best_rmse = float('inf')
ckpt = os.path.join(params.result_dir, 'best_ctx9.pt')

pbar = tqdm(range(params.total_steps), desc="Train(ctx9)")
run_loss = 0.0
it = iter(train_loader)

for step in pbar:
    try:
        x0, msk, cond, mu, std = next(it)
    except StopIteration:
        it = iter(train_loader)
        x0, msk, cond, mu, std = next(it)

    x0   = x0.to(DEVICE, non_blocking=True)
    msk  = msk.to(DEVICE, non_blocking=True)
    cond = cond.to(DEVICE, non_blocking=True)

    # DEBUG PRINT
    if step == 0:
        print("[Train] shapes:",
            "x0", tuple(x0.shape),
            "msk", tuple(msk.shape),
            "cond", tuple(cond.shape))

    # 주기적으로 x0 포화율 확인
    if (step+1) % params.log_every == 0:
        with torch.no_grad():
            sat = (x0.abs() > 0.98).float().mean().item()
            print(f"[Train] step={step+1} x0 range=({x0.min().item():.2f},{x0.max().item():.2f}) "
                f"sat(|x0|>0.98)={sat:.3f} | cond range=({cond.min().item():.2f},{cond.max().item():.2f}) "
                f"mask mean={msk.mean().item():.3f}")

    opt.zero_grad(set_to_none=True)
    loss = diffusion(x0, msk, cond)
    loss.backward()
    nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
    opt.step()
    sched.step()

    run_loss += loss.item()
    if (step+1) % params.log_every == 0:
        avg = run_loss / params.log_every
        run_loss = 0.0
        # grab LR robustly
        curr_lr = opt.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{curr_lr:.2e}")


    if (step+1) % params.eval_every == 0:
        diffusion.eval()
        with torch.inference_mode():
            # rmse_raw, rmse_norm, nobs = eval_rmse(diffusion, test_loader) # FIXME : original eval code
            rmse_raw, rmse_norm, nobs = eval_rmse_with_pbar(diffusion, test_loader, max_batches=10, show_pbar=True)
            # rmse_raw, rmse_norm, nobs = eval_rmse_minibatch(diffusion, test_loader, max_batches=2)
            # m = eval_crps_and_points(diffusion, test_loader, K=10, clip=False)  # <-- CRPS (and friends)
        # print(f"\n[Step {step+1}] Test RMSE_raw={rmse_raw:.3f} µg/m³ | RMSE_norm={rmse_norm:.3f} (over {nobs} bins)")
        print(f"\n[Step {step+1}] Test RMSE_raw={rmse_raw:.3f} µV (over {nobs} bins)")
        
        # print(f"\n[Step {step+1}] Test RMSE_raw={rmse_raw:.3f} µg/m³ | RMSE_norm={rmse_norm:.3f} "
        #       f"(over {nobs} bins) | CRPS_raw={m['CRPS_raw']:.3f} µg/m³ (K={m['K']})")
        if rmse_raw < best_rmse:
            best_rmse = rmse_raw
            torch.save({'unet': unet.state_dict(),
                        'diff': diffusion.state_dict(),
                        'H': H, 'W': W}, ckpt)
            print(f"  >> Saved best checkpoint @ {ckpt} (RMSE_raw={best_rmse:.3f})")
        diffusion.train()

    if (step+1) % params.save_samples_every == 0:
        diffusion.eval()
        with torch.inference_mode():
            # grab a test batch, sample, and save plain grids (tanh -> [0,1] for viewing)
            xb, mb, cb, mub, stdb = next(iter(test_loader))
            xhat = diffusion.sample(cb.to(DEVICE)).cpu()
            t0 = xhat.size(1)//2
            xhat2d = xhat[:, t0]
            vis = (xhat2d.unsqueeze(1) + 1.0) * 0.5
            save_image(vis, os.path.join(params.result_dir, f"samples_step{step+1}.png"), nrow=4)
        diffusion.train()

print("Done. Best Test RMSE_raw:", best_rmse)

# ============================================================
# 9) Inference + overlay: GT vs Pred (only observed bins)
# ============================================================
def overlay_panel(test_loader, model, save_path=os.path.join(params.result_dir, 'ctx9_overlay.png'),
                  back_img='./airdelhi_background.png'):
    try:
        back = plt.imread(back_img)
    except:
        back = None

    import matplotlib.colors as colors
    from matplotlib.colors import LinearSegmentedColormap
    cmap0 = LinearSegmentedColormap.from_list('', ['white', 'orange', 'red'])
    cmap0 = cmap0.copy()
    cmap0.set_bad(color='lightgray')

    with torch.inference_mode():
        xb, mb, cb, mub, stdb = next(iter(test_loader))
        xhat = model.sample(cb.to(DEVICE), clip=True).cpu()  # tanh(z)

        if params.squash_tanh:
            raw_hat = inv_tanh_to_raw(xhat, mub[:,None,None,None], stdb[:,None,None,None]).squeeze(1)
            raw_gt  = inv_tanh_to_raw(xb,   mub[:,None,None,None], stdb[:,None,None,None]).squeeze(1)
        else:
            raw_hat = xhat.squeeze(1)
            raw_hat = raw_hat*params.data_scaling_factor
            raw_gt = xb.squeeze(1)
            raw_gt = raw_gt*params.data_scaling_factor

        # t0 = xhat.size(1)//2
        # raw_hat = raw_hat[:,t0]
        # raw_gt = raw_gt[:,t0]
        mb      = mb.squeeze(1)

    # vmax = float(mub.mean() + 3*stdb.mean())
    vmax = np.nanpercentile(raw_gt.numpy(), 99)
    lon_edges = np.linspace(0,1,W+1); lat_edges = np.linspace(0,1,H+1)

    B = min(8, raw_hat.size(0))
    fig, axes = plt.subplots(2, B, figsize=(3.4*B, 6.8))
    for i in range(B):
        for row, arr in enumerate([raw_gt[i].numpy(), raw_hat[i].numpy()]):
            ax = axes[row, i]
            if back is not None: ax.imshow(back, extent=[0,1,0,1], alpha=0.6)
            mask = (mb[i].numpy() == 0)
            arr_plot = np.ma.array(arr, mask=mask)
            # arr_plot = arr.copy()
            # arr_plot[mb[i].numpy() == 0] = np.nan  # show only observed bins
            pm = ax.pcolormesh(lon_edges, lat_edges, arr_plot, cmap=cmap0,
                               norm=colors.Normalize(vmin=0, vmax=vmax))
            # draw grid
            for y in lat_edges: ax.plot([0,1],[y,y], c='k', lw=0.1)
            for x in lon_edges: ax.plot([x,x],[0,1], c='k', lw=0.1)
            ax.set_axis_off()
        axes[0, i].set_title("GT (obs bins)", fontsize=11)
        axes[1, i].set_title("Pred (obs bins)", fontsize=11)

    cbar = fig.colorbar(pm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout(); plt.savefig(save_path, dpi=140, bbox_inches='tight'); plt.close(fig)
    print("Saved overlays to:", save_path)

import os, math
import numpy as np
import matplotlib.pyplot as plt
import torch

def timecurve_panel(
    test_loader, model,
    save_path=os.path.join(params.result_dir, "timecurve_overlay.png"),
    sample_idx=0,              # 배치에서 몇 번째 샘플을 볼지
    k_points=6,                # (h,w) 몇 개 찍을지
    points=None,               # [(h,w), ...] 직접 지정하면 이걸 사용
    pick="random",             # points=None일 때: "random" or "maxvar"
    seed=41,
):
    """
    GT(파란색) vs Pred(오렌지색) time series overlay plot.
    raw_hat/raw_gt assumed to be (B,T,H,W) after squeeze(1) etc.
    """

    with torch.inference_mode():
        xb, mb, cb, mub, stdb = next(iter(test_loader))
        xhat = model.sample(cb.to(DEVICE), clip=True).cpu()  # tanh(z)

        # --- raw로 복원 ---
        if params.squash_tanh:
            raw_hat = inv_tanh_to_raw(xhat, mub[:,None,None,None], stdb[:,None,None,None]).squeeze(1)
            raw_gt  = inv_tanh_to_raw(xb,   mub[:,None,None,None], stdb[:,None,None,None]).squeeze(1)
        else:
            raw_hat = xhat.squeeze(1)
            raw_hat = raw_hat*params.data_scaling_factor
            raw_gt = xb.squeeze(1)
            raw_gt = raw_gt*params.data_scaling_factor

        # raw_hat/raw_gt: (B,T,H,W) 라고 가정
        # mb: (B,1,H,W) 또는 (B,T,H,W)일 수 있음 → (B,*,H,W)로 맞춰 사용
        mb = mb.squeeze(1)  # (B,H,W) or (B,T,H,W)

    B, T, Hh, Ww = raw_hat.shape
    i = int(sample_idx)
    assert 0 <= i < B, f"sample_idx out of range: {i} (B={B})"

    # --- 관측 위치 후보 만들기 ---
    # mb가 (B,H,W)이면 그 마스크를 그대로 쓰고
    # mb가 (B,T,H,W)이면 "한 번이라도 관측된" 위치를 후보로 쓰자(보수적)
    if mb.dim() == 3:
        obs_mask = mb[i].cpu().numpy().astype(bool)          # (H,W)
    elif mb.dim() == 4:
        obs_mask = (mb[i].sum(dim=0) > 0).cpu().numpy().astype(bool)  # (H,W)
    else:
        raise ValueError(f"Unexpected mb shape after squeeze: {mb.shape}")

    cand = np.argwhere(obs_mask)  # [(h,w),...]
    if cand.size == 0:
        raise RuntimeError("No observed (h,w) candidates found from mask.")

    # --- 찍을 points 선택 ---
    if points is None:
        rng = np.random.default_rng(seed)

        if pick == "random":
            sel = cand[rng.choice(len(cand), size=min(k_points, len(cand)), replace=False)]
            points = [(int(h), int(w)) for h, w in sel]

        elif pick == "maxvar":
            # GT time series variance가 큰 위치 위주로 고르기 (패턴 비교가 더 잘 보임)
            gt_i = raw_gt[i].cpu().numpy()  # (T,H,W)
            vars_ = []
            for (h, w) in cand:
                ts = gt_i[:, h, w]
                vars_.append(np.var(ts))
            vars_ = np.array(vars_)
            top_idx = np.argsort(-vars_)[:min(k_points, len(cand))]
            sel = cand[top_idx]
            points = [(int(h), int(w)) for h, w in sel]
        else:
            raise ValueError("pick must be 'random' or 'maxvar'")
    else:
        # 사용자 지정 points 검증
        points = [(int(h), int(w)) for (h, w) in points]
        for (h, w) in points:
            if not (0 <= h < Hh and 0 <= w < Ww):
                raise ValueError(f"Point {(h,w)} out of bounds for (H,W)=({Hh},{Ww})")

    # --- plot ---
    K = len(points)
    ncol = min(3, K)
    nrow = int(math.ceil(K / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2*ncol, 3.6*nrow), sharex=True)
    axes = np.array(axes).reshape(-1)

    t = np.arange(T)
    gt_i  = raw_gt[i].cpu().numpy()   # (T,H,W)
    hat_i = raw_hat[i].cpu().numpy()  # (T,H,W)

    for ax_idx, (h, w) in enumerate(points):
        ax = axes[ax_idx]
        gt_ts  = gt_i[:, h, w]
        hat_ts = hat_i[:, h, w]

        ax.plot(t, gt_ts,  color="tab:blue",   lw=2.0, label="GT")
        ax.plot(t, hat_ts, color="tab:orange", lw=2.0, label="Pred", alpha=0.9)

        ax.set_title(f"sample {i} | (h,w)=({h},{w})", fontsize=11)
        ax.set_xlabel("t")
        ax.set_ylabel("raw value")
        ax.grid(True, lw=0.3, alpha=0.5)

        # 각 subplot마다 legend 넣으면 지저분해서 첫 plot에만
        if ax_idx == 0:
            ax.legend(frameon=False, fontsize=10)

    # 남는 축 숨기기
    for j in range(K, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved time-curve overlays to:", save_path)


# ---- Run an overlay on current (or best-loaded) model ----
overlay_panel(test_loader, diffusion)
# timecurve_panel(test_loader, diffusion, points=[(4,6), (6,4), (8,6), (6,8)])

# ============================================================
# 10) Load best checkpoint & run full test-day eval again
# ============================================================
def load_and_eval(ckpt_path, test_loader):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    unet = UNet(base_dim=128, dim_mults=(1,2,4), in_channels=IN_CHANNELS, image_size=(H,W)).to(DEVICE)
    unet.load_state_dict(ck['unet'])
    diff = GaussianDiffusion(unet, image_size=(H,W), time_steps=params.time_steps, loss_type='l2').to(DEVICE)
    diff.load_state_dict(ck['diff'])
    diff.eval()

    # rmse_raw, rmse_norm, nobs = eval_rmse(diff, test_loader) # FIXME : original eval code
    rmse_raw, rmse_norm, nobs = eval_rmse_with_pbar(diffusion, test_loader, max_batches=1000, show_pbar=True)
    # rmse_raw, rmse_norm, nobs = eval_rmse_minibatch(diff, test_loader, max_batches=2)
    # print(f"[BEST] Test RMSE_raw={rmse_raw:.3f} µg/m³ | RMSE_norm={rmse_norm:.3f} (over {nobs} bins)")
    print(f"[BEST] Test RMSE_raw={rmse_raw:.3f} µV (over {nobs} bins)")
    overlay_panel(test_loader, diff, save_path=os.path.join(params.result_dir, 'ctx9_overlay_best.png'))
    # timecurve_panel(test_loader, diffusion, points=[(4,6), (6,4), (8,6), (6,8)])
    return diff

# Example (after training): 
diff_best = load_and_eval(ckpt,test_loader)