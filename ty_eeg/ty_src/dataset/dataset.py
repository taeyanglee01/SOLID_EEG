import torch
import lmdb
import pickle

from torch.utils.data import Dataset, DataLoader

from .utils.utils import random_seed


class Physio_1sec_raw_for_SOLID_from_lmdb(Dataset):
    def __init__(
            self,
            lmdb_dir: str,
            maxfold: int,
            targetfold: int,
            seed: int,
            train: bool,
            split_by_sub: bool,
            seg_len_pts: int = 10,
            stride_pts: int = 10,
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
            # x0 = x0 / 3
            x0 = torch.tanh(z)
        else:
            x0 = full_grid/params.data_scaling_factor  # (L,H,W)

        # DEBUG PRINT
        if self.squash and idx == 0:
            with torch.no_grad():
                z = (full_grid - self.mean) / (self.std + 1e-6)
                sat = (x0.abs() > 0.98).float().mean().item()
                '''
                print(f"[DS] mean={self.mean:.4f} std={self.std:.4f} | "
                    f"z: min={z.min():.2f} max={z.max():.2f} p(|z|>3)={(z.abs()>3).float().mean():.3f} | "
                    f"x0: min={x0.min():.2f} max={x0.max():.2f} sat(|x0|>0.98)={sat:.3f}")
                '''
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

class OneInstanceDataset(Dataset):
    def __init__(self, base_dataset, fixed_idx=0):
        self.base = base_dataset
        self.fixed_idx = fixed_idx

        # base에 있는 통계도 그대로 노출 (EEGToGridCtx9_1sec에서 getattr로 읽음)
        self.mean = float(getattr(base_dataset, "mean", 0.0))
        self.std  = float(getattr(base_dataset, "std", 1.0))

    def __len__(self):
        return 1

    def __getitem__(self, _):
        return self.base[self.fixed_idx]    