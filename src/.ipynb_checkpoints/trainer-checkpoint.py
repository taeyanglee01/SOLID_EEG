import os
import math
import numpy as np
from .dataset import dataset_wrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
from torchvision.utils import save_image, make_grid
from multiprocessing import cpu_count
from tqdm import tqdm
import datetime
from termcolor import colored
from .utils import *
from .sparsity import SparsityController
import torch.nn.functional as F
from typing import Optional
from scipy.interpolate import RBFInterpolator  # <-- needed for interpolate modes
from collections import OrderedDict
from scipy.ndimage import distance_transform_edt



def cycle(dl):
    while True:
        for data in dl:
            yield data

def _atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(min=-1+eps, max=1-eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    
@torch.no_grad()
def rbf_interpolate_dense(x_unit: torch.Tensor,
                          m: torch.Tensor,
                          kernel: str = "thin_plate_spline",
                          epsilon: Optional[float] = None,
                          neighbors: Optional[int] = 200,
                          smoothing: float = 1e-2):
    """
    RBF interpolation that operates in z-space for stability:
      unit [-1,1] --atanh--> z  --RBF--> z_hat  --tanh--> unit_hat

    x_unit: [B,C,H,W] (unit range)
    m     : [B,C,H,W] binary mask (1=known)
    """
    assert x_unit.ndim == 4 and m.shape == x_unit.shape
    device = x_unit.device
    B, C, H, W = x_unit.shape

    # Work in z-space
    x_z = _atanh_clamped(x_unit)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, H, dtype=np.float64),
        np.linspace(0.0, 1.0, W, dtype=np.float64),
        indexing="ij"
    )
    grid = np.stack([yy, xx], axis=-1).reshape(-1, 2)  # [H*W, 2]

    # Heuristic epsilon if not provided: ~3 pixels in [0,1] coords
    if epsilon is None and kernel != "thin_plate_spline":
        epsilon = 3.0 / float(max(H, W))

    x_np = x_z.detach().cpu().numpy().astype(np.float64)
    m_np = (m.detach().cpu().numpy() > 0.5)

    out_z = np.empty_like(x_np)

    for b in range(B):
        for c in range(C):
            known = m_np[b, c].reshape(-1)
            if not known.any():
                # no support -> zeros (in z), which becomes 0 in unit space
                out_z[b, c] = np.zeros((H, W), dtype=np.float64)
                continue

            pts  = grid[known]                                  # [N,2]
            vals = x_np[b, c].reshape(-1)[known][:, None]       # [N,1]

            rbf = RBFInterpolator(
                pts, vals, kernel=kernel,
                epsilon=None if kernel == "thin_plate_spline" else float(epsilon),
                neighbors=neighbors, smoothing=float(smoothing)
            )
            pred = rbf(grid)[:, 0].reshape(H, W)

            # keep exact GT on knowns (still in z-space)
            pred[m_np[b, c]] = x_np[b, c][m_np[b, c]]
            out_z[b, c] = pred

    out_z_t = torch.from_numpy(out_z).to(device).type_as(x_unit)
    # back to unit space
    return torch.tanh(out_z_t)


@torch.no_grad()
def nn_interpolate_dense(x: torch.Tensor, m: torch.Tensor):
    """
    Nearest-neighbor 'inpainting' using EDT (Voronoi fill).
    x: [B,C,H,W]
    m: [B,C,H,W] binary (1=known). Returns dense [B,C,H,W].
    """
    assert x.ndim == 4 and m.shape == x.shape
    device = x.device
    B, C, H, W = x.shape

    x_np = x.detach().cpu().numpy()
    m_np = (m.detach().cpu().numpy() > 0.5)
    out  = np.empty_like(x_np)

    for b in range(B):
        for c in range(C):
            known = m_np[b, c]  # True where we know the value
            if not known.any():
                out[b, c] = np.zeros((H, W), dtype=x_np.dtype)
                continue

            # Make KNOWN pixels the "background" (zeros) so EDT points to nearest known.
            inv = ~known

            # IMPORTANT: only request indices (no distances) and cast to integer
            inds = distance_transform_edt(
                inv, return_indices=True, return_distances=False
            )                                 # shape: (2, H, W)
            rows, cols = inds[0].astype(np.intp), inds[1].astype(np.intp)

            out[b, c] = x_np[b, c][rows, cols]
            # keep exact GT on knowns (usually a no-op)
            out[b, c][known] = x_np[b, c][known]

    return torch.from_numpy(out).to(device).type_as(x)


        
class Trainer:
    def __init__(self, diffusion_model, dataset, batch_size=32, lr=2e-5, total_step=100000, ddim_samplers=None,
                 save_and_sample_every=1000, num_samples=25, result_folder='./results', cpu_percentage=0,
                 fid_estimate_batch_size=None, ddpm_fid_score_estimate_every=None, ddpm_num_fid_samples=None,
                 max_grad_norm=1., tensorboard=False, exp_name=None, clip=True,
                 sparsity_mode='random_epoch', sparsity_pattern='random', sparsity_level=0.2, block_size=5, num_blocks=5,
                 mask_mode='cond_oo', **dataset_kwargs):

        now = datetime.datetime.now()
        self.cur_time = now.strftime('%Y-%m-%d_%Hh%Mm')
        if exp_name is None:
            exp_name = os.path.basename(dataset)
            if exp_name == '':
                exp_name = os.path.basename(os.path.dirname(dataset))
        self.exp_name = exp_name
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers or []
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.nrow = int(math.sqrt(self.num_samples))
        assert (self.nrow ** 2) == self.num_samples, 'num_samples must be a square number. ex) 25, 36, 49, ...'
        self.save_and_sample_every = save_and_sample_every
        self.image_size = self.diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.result_folder = os.path.join(result_folder, exp_name, self.cur_time)
        self.ddpm_result_folder = os.path.join(self.result_folder, 'DDPM')
        self.device = self.diffusion_model.device
        self.clip = clip
        self.ddpm_fid_flag = True if ddpm_fid_score_estimate_every is not None else False
        self.ddpm_fid_score_estimate_every = ddpm_fid_score_estimate_every
        self.cal_fid = True if self.ddpm_fid_flag else False
        self.tqdm_sampler_name = None
        self.tensorboard = tensorboard
        self.tensorboard_name = None
        self.writer = None
        self.global_step = 0
        self.total_step = total_step
        self.fid_score_log = dict()
        self.mask_mode = (mask_mode or 'cond_oo')


        ### --- RBF cache state ---
        self._rbf_cache_enabled = (
    (sparsity_mode == 'fixed_instance')
    and (
        str(mask_mode).startswith('interp')
        or str(mask_mode).startswith('nn')
        or mask_mode in ('both', 'interp_oo', 'interp_xo', 'interp_ox',
                         'nn_oo', 'nn_xo', 'nn_ox')
    )
)
        self._rbf_cache_max = 20000          # tune to your dataset size / RAM
        self._rbf_cache = OrderedDict()      # keys: (sample_id:int, kind:str['x'|'y'])
        # ----------------------------------

            
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if clip is True or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'clip'), exist_ok=True)
        if clip is False or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'no_clip'), exist_ok=True)

        # ===== Dataset & DataLoader & Optimizer =====
        print(make_notification('Dataset', color='green'))
        dataSet = dataset_wrapper(dataset, self.image_size, **dataset_kwargs)
        assert len(dataSet) >= 100, 'you should have at least 100 images in your folder.at least 10k images recommended'
        print(colored(f'Dataset Length: {len(dataSet)}\n', 'green'))
        CPU_cnt = cpu_count()
        num_workers = int(CPU_cnt * cpu_percentage)
        assert num_workers <= CPU_cnt, "cpu_percentage must be [0.0, 1.0]"
        dataLoader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
        self.dataLoader = cycle(dataLoader)           # <-- always cycle; dataset returns 3-tuple
        self.optimizer = Adam(self.diffusion_model.parameters(), lr=lr)

        self.sparsity_controller = SparsityController(
            image_size=self.image_size,
            mode=sparsity_mode,
            pattern=sparsity_pattern,
            sparsity=sparsity_level,
            block_size=block_size,
            num_blocks=num_blocks
        )

        # ===== DDIM sampler setting =====
        self.ddim_sampling_schedule = list()
        for idx, sampler in enumerate(self.ddim_samplers):
            sampler.sampler_name = f'DDIM_{idx + 1}_steps{sampler.ddim_steps}_eta{sampler.eta}'
            self.ddim_sampling_schedule.append(sampler.sample_every)
            save_path = os.path.join(self.result_folder, sampler.sampler_name)
            sampler.save_path = save_path
            if sampler.save:
                os.makedirs(save_path, exist_ok=True)
            if sampler.generate_image:
                if sampler.clip is True or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'clip'), exist_ok=True)
                if sampler.clip is False or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'no_clip'), exist_ok=True)
            if sampler.calculate_fid:
                self.cal_fid = True
                if self.tqdm_sampler_name is None:
                    self.tqdm_sampler_name = sampler.sampler_name
                sampler.num_fid_sample = sampler.num_fid_sample if sampler.num_fid_sample is not None else len(dataSet)
                self.fid_score_log[sampler.sampler_name] = list()
            if getattr(sampler, "fixed_noise", False):
                sampler.register_buffer('noise', torch.randn([self.num_samples, sampler.channel,
                                                              sampler.image_size, sampler.image_size]))

        # ===== Image generation log =====
        print(make_notification('Image Generation', color='cyan'))
        print(colored(f'-> DDPM Sampler / Image generation every {self.save_and_sample_every} steps', 'cyan'))
        for sampler in self.ddim_samplers:
            if sampler.generate_image:
                print(colored(f'-> {sampler.sampler_name} / Image generation every {sampler.sample_every} steps / Fixed Noise : {sampler.fixed_noise}', 'cyan'))
        print('\n')

        # ===== FID score =====
        print(make_notification('FID', color='magenta'))
        if not self.cal_fid or dataset.lower() == 'navierstokes':
            print(colored('No FID evaluation will be executed!\n'
                          'If you want FID evaluation consider using DDIM sampler.', 'magenta'))
        else:
            self.fid_batch_size = fid_estimate_batch_size if fid_estimate_batch_size is not None else self.batch_size
            dataSet_fid = dataset_wrapper(dataset, self.image_size,
                                          augment_horizontal_flip=False, info_color='magenta', min1to1=False,
                                          **dataset_kwargs)
            dataLoader_fid = DataLoader(dataSet_fid, batch_size=self.fid_batch_size, num_workers=num_workers)
            self.fid_scorer = FID(self.fid_batch_size, dataLoader_fid, dataset_name=exp_name, device=self.device,
                                  no_label=os.path.isdir(dataset))
            print(colored('FID score will be calculated with the following sampler(s)', 'magenta'))
            if self.ddpm_fid_flag:
                self.ddpm_num_fid_samples = ddpm_num_fid_samples if ddpm_num_fid_samples is not None else len(dataSet)
                print(colored(f'-> DDPM Sampler / FID calculation every {self.ddpm_fid_score_estimate_every} steps with {self.ddpm_num_fid_samples} generated samples', 'magenta'))
            for sampler in self.ddim_samplers:
                if sampler.calculate_fid:
                    print(colored(f'-> {sampler.sampler_name} / FID calculation every {sampler.sample_every} steps with {sampler.num_fid_sample} generated samples', 'magenta'))
            print('\n')
            if self.ddpm_fid_flag:
                self.tqdm_sampler_name = 'DDPM'
                self.fid_score_log['DDPM'] = list()
                print(make_notification('WARNING', color='red', boundary='*'))
                msg = (
                    "FID computation with DDPM sampler requires a lot of generated samples and can therefore be very time "
                    "consuming.\nTo accelerate sampling, only using DDIM sampling is recommended. To disable DDPM sampling, "
                    "set [ddpm_fid_score_estimate_every] parameter to None while instantiating Trainer.\n"
                )
                print(colored(msg, 'red'))
            del dataLoader_fid
            del dataSet_fid


    @torch.no_grad()
    def _rbf_interpolate_dense_single(self, x_bchw: torch.Tensor, m_bchw: torch.Tensor) -> torch.Tensor:
        """
        x_bchw, m_bchw: shape [1,1,H,W] on device. Returns [1,1,H,W] on device.
        Uses your existing numpy+SciPy function internally.
        """
        return rbf_interpolate_dense(x_bchw, m_bchw, kernel="gaussian", epsilon=None, neighbors=200, smoothing=1e-2)
        # or try: kernel="thin_plate_spline", neighbors=200, smoothing=1e-2

    
    ### NEW: cached batch helper
    @torch.no_grad()
    def _dense_rbf_cached_batch(self, img: torch.Tensor, mask: torch.Tensor, sample_ids, kind: str) -> torch.Tensor:
        """
        img, mask: [B,1,H,W] on device. Returns [B,1,H,W] on device.
        Caches CPU copies per (sample_id, kind).
        """
        B = img.shape[0]
        outs = []
        for b in range(B):
            sid = int(sample_ids[b])
            key = (sid, kind)
            cached = self._rbf_cache.get(key, None) if self._rbf_cache_enabled else None
            if cached is not None:
                # LRU update
                self._rbf_cache.move_to_end(key)
                outs.append(cached.to(img.device, non_blocking=True).type_as(img))
            else:
                out_b = self._rbf_interpolate_dense_single(img[b:b+1], mask[b:b+1])
                outs.append(out_b)
                if self._rbf_cache_enabled:
                    # store CPU to save VRAM
                    self._rbf_cache[key] = out_b.detach().cpu()
                    self._rbf_cache.move_to_end(key)
                    # LRU eviction
                    if len(self._rbf_cache) > self._rbf_cache_max:
                        self._rbf_cache.popitem(last=False)
        return torch.cat(outs, dim=0)




    @torch.no_grad()
    def _dense_nn_cached_batch(self, img: torch.Tensor, mask: torch.Tensor, sample_ids, kind: str) -> torch.Tensor:
        B = img.shape[0]
        outs = []
        for b in range(B):
            sid = int(sample_ids[b])
            key = (sid, f"nn_{kind}")
            cached = self._rbf_cache.get(key, None) if self._rbf_cache_enabled else None
            if cached is not None:
                self._rbf_cache.move_to_end(key)
                outs.append(cached.to(img.device, non_blocking=True).type_as(img))
            else:
                out_b = nn_interpolate_dense(img[b:b+1], mask[b:b+1])
                outs.append(out_b)
                if self._rbf_cache_enabled:
                    self._rbf_cache[key] = out_b.detach().cpu()
                    self._rbf_cache.move_to_end(key)
                    if len(self._rbf_cache) > self._rbf_cache_max:
                        self._rbf_cache.popitem(last=False)
        return torch.cat(outs, dim=0)

    def train(self):
        print(make_notification('Training', color='yellow', boundary='+'))
        cur_fid = 'NAN'
        ddpm_best_fid = 1e10
        stepTQDM = tqdm(range(self.global_step, self.total_step))

        running_loss = 0.0
        running_count = 0
        last_avg_loss = None

        for cur_step in stepTQDM:
            self.diffusion_model.train()
            self.optimizer.zero_grad()

            # --------- Forecasting batch: (x_t, y_{t+1}, sample_id) ----------
            x, y, sample_id = next(self.dataLoader)
            x = x.to(self.device, non_blocking=True).float()   # (B,1,H,W) previous frame
            y = y.to(self.device, non_blocking=True).float()   # (B,1,H,W) next frame (target)
            B, C, H, W = y.shape
            assert C == 1, f"Expected single-channel NS image, got {C}"
            sample_ids = [int(s.item()) for s in sample_id]

            # --------- H/T masks (floats) ----------
            cond_masks, target_masks = self.sparsity_controller.get_masks(B, C, sample_ids)
            Hmask = torch.stack(cond_masks).to(self.device).float()   # (B,1,H,W)
            Tmask = torch.stack(target_masks).to(self.device).float()

            # Bool for logic
            Hmask_bool = Hmask > 0.5
            Tmask_bool = Tmask > 0.5
            if torch.logical_and(Hmask_bool, Tmask_bool).any():
                raise RuntimeError("H and T masks overlap; fix SparsityController.")

            # Convenience
            def _dense_zero(img, m):   # keep values on m, zeros elsewhere
                return img * m
            def _dense_rbf(img, m):
                # z-space + adaptive epsilon + local neighbors + smoothing
                return rbf_interpolate_dense(img, m, kernel="thin_plate_spline", epsilon=None, neighbors=200, smoothing=1e-2)



            def _dense_rbf_x(img, m):  # interpolate x_t with Hmask
                return self._dense_rbf_cached_batch(img, m, sample_ids, kind='x')
            def _dense_rbf_y(img, m):  # interpolate y_{t+1} with Tmask
                return self._dense_rbf_cached_batch(img, m, sample_ids, kind='y')

            def _dense_nn_x(img, m):  # NN on x_t with Hmask
                return self._dense_nn_cached_batch(img, m, sample_ids, kind='x')
            
            def _dense_nn_y(img, m):  # NN on y_{t+1} with Tmask
                return self._dense_nn_cached_batch(img, m, sample_ids, kind='y')


            # Normalize mask_mode names (support old *_rbf suffix)
            mm = (self.mask_mode or 'cond_oo').lower()
            if mm.endswith('_rbf'):
                mm = mm[:-4]
            if mm in ('both',):
                mm = 'cond_oo'

            # ===== Default forecasting (cond O O) =====
            x_in_for_model      = x * Hmask      # condition on sparse previous frame
            cond_mask_for_model = Hmask          # tell model where x_t is known
            target_for_loss     = y              # learn true next frame
            loss_mask_for_model = Tmask          # grade on held-out next-step pixels
            x0_for_model        = target_for_loss

            # ===== Routing variants (expressed in terms of x (input) / y (target)) =====
            if mm == 'no_proc_xx':
                # Input: zeros off-H; tell model there is NO mask (X); Target: zeros off-T; full-grid loss
                x_in_for_model      = _dense_zero(x, Hmask)
                cond_mask_for_model = torch.zeros_like(Hmask)
                target_for_loss     = _dense_zero(y, Tmask)
                loss_mask_for_model = None
                x0_for_model        = target_for_loss

            elif mm == 'interp_ox':
                # Input: RBF-dense from H; pretend dense (O); Target: zeros off-T; full-grid loss
                x_in_for_model      = _dense_rbf(x, Hmask)
                cond_mask_for_model = torch.ones_like(Hmask)
                target_for_loss     = _dense_zero(y, Tmask)
                loss_mask_for_model = None
                x0_for_model        = target_for_loss

            elif mm == 'interp_xo':
                # Input: zeros off-H; hide mask (X); Target: RBF-dense from T; full-grid loss
                x_in_for_model      = _dense_zero(x, Hmask)
                cond_mask_for_model = torch.zeros_like(Hmask)
                target_for_loss     = _dense_rbf(y, Tmask)
                loss_mask_for_model = None
                x0_for_model        = target_for_loss

            elif mm == 'interp_oo':
                # Input: RBF-dense; say dense (O); Target: RBF-dense; full-grid loss
                x_in_for_model      = _dense_rbf_x(x, Hmask)   ### cached
                cond_mask_for_model = torch.ones_like(Hmask)
                target_for_loss     = _dense_rbf_y(y, Tmask)   ### cached
                loss_mask_for_model = None
                x0_for_model        = target_for_loss


            elif mm == 'cond_ox':
                # Input: sparse x; expose mask (O); Target: zeros off-T; full-grid loss (to zeros)
                x_in_for_model      = x * Hmask
                cond_mask_for_model = Hmask
                target_for_loss     = _dense_zero(y, Tmask)
                loss_mask_for_model = None
                x0_for_model        = target_for_loss

            elif mm == 'cond_xo':
                # Input: sparse x; hide mask (X); Target: full y; loss only on T
                x_in_for_model      = x * Hmask
                cond_mask_for_model = torch.zeros_like(Hmask)
                target_for_loss     = y
                loss_mask_for_model = Tmask
                x0_for_model        = target_for_loss

            elif mm == 'nn_oo':
                # Input: NN-dense; say dense (O); Target: NN-dense; full-grid loss
                x_in_for_model      = _dense_nn_x(x, Hmask)
                cond_mask_for_model = torch.ones_like(Hmask)
                target_for_loss     = _dense_nn_y(y, Tmask)
                loss_mask_for_model = None
                x0_for_model        = target_for_loss

            elif mm == 'cond_oo':
                # Baseline forecasting: sparse x + exposed mask; full y; loss on T
                # (already set by defaults)
                pass

            else:
                raise ValueError(f"Unknown mask_mode='{self.mask_mode}'. Expected one of: "
                                 f"no_proc_xx, interp_ox, interp_xo, interp_oo, cond_ox, cond_xo, cond_oo, both, *_rbf aliases.")

            # ---------- Train step ----------
            loss = self.diffusion_model(
                y,
                sparse_input=x_in_for_model,
                perceiver_input=None,
                mask=cond_mask_for_model,
                loss_mask=loss_mask_for_model
            )

            running_loss += loss.item()
            running_count += 1

            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if running_count % 750 == 0:
                last_avg_loss = running_loss / running_count
                running_loss = 0.0
                running_count = 0

            vis_fid = cur_fid if isinstance(cur_fid, str) else f'{cur_fid:.04f}'
            postfix = {'loss': f'{loss.item():.04f}', 'FID': vis_fid, 'step': self.global_step}
            if last_avg_loss is not None:
                postfix['avg_loss'] = f'{last_avg_loss:.6f}'
            stepTQDM.set_postfix(postfix)

            self.diffusion_model.eval()
            # ===== DDPM Sampler for generating images =====
            if cur_step != 0 and (cur_step % self.save_and_sample_every) == 0:
                if self.writer is not None:
                    self.writer.add_scalar('Loss', loss.item(), cur_step)
                with torch.inference_mode():
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    for i, j in zip([True, False], ['clip', 'no_clip']):
                        if self.clip not in [i, 'both']:
                            continue

                        imgs = []
                        start_idx = 0
                        for n in batches:
                            mask_batch    = cond_mask_for_model[start_idx:start_idx+n]
                            sparse_batch  = x_in_for_model[start_idx:start_idx+n]

                            sampled = self.diffusion_model.sample(
                                batch_size=n,
                                sparse_input=sparse_batch,
                                perceiver_input=None,
                                mask=mask_batch,
                                clip=i
                            )
                            imgs.append(sampled)
                            start_idx += n

                        imgs = torch.cat(imgs, dim=0)
                        save_image(imgs, nrow=self.nrow,
                                   fp=os.path.join(self.ddpm_result_folder, j, f'sample_{cur_step}.png'))
                        if self.writer is not None:
                            self.writer.add_images(f'DDPM sampling result ({j})', imgs, cur_step)

                        # Simple colormapped composite
                        def apply_colormap(tensor, cmap_name='viridis'):
                            import matplotlib.pyplot as plt
                            cmap = plt.get_cmap(cmap_name)
                            tensor_np = tensor.detach().cpu().numpy()
                            tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min() + 1e-8)
                            rgb_list = []
                            for img in tensor_np:
                                img_2d = img[0]
                                rgba_img = cmap(img_2d)
                                rgb_img = rgba_img[..., :3]
                                rgb_list.append(torch.from_numpy(rgb_img).permute(2, 0, 1))
                            return torch.stack(rgb_list, dim=0)

                        y_vis      = apply_colormap(y[:8])
                        sparse_vis = apply_colormap(x_in_for_model[:8])
                        mask_vis   = apply_colormap(cond_mask_for_model[:8])
                        imgs_vis   = apply_colormap(imgs[:8])
                        Tmask_vis  = apply_colormap(Tmask[:8])

                        viz = torch.cat([y_vis, sparse_vis, mask_vis, imgs_vis, Tmask_vis], dim=0)
                        grid = make_grid(viz, nrow=self.nrow)
                        save_image(grid, os.path.join(self.ddpm_result_folder, j, f'composite_{cur_step}_viridis.png'))

                self.save('latest')

            # ===== DDPM FID =====
            if self.ddpm_fid_flag and cur_step != 0 and (cur_step % self.ddpm_fid_score_estimate_every) == 0:
                ddpm_cur_fid, _ = self.fid_scorer.fid_score(self.diffusion_model.sample, self.ddpm_num_fid_samples)
                if ddpm_best_fid > ddpm_cur_fid:
                    ddpm_best_fid = ddpm_cur_fid
                    self.save('best_fid_ddpm')
                if self.writer is not None:
                    self.writer.add_scalars('FID', {'DDPM': ddpm_cur_fid}, cur_step)
                cur_fid = ddpm_cur_fid
                self.fid_score_log['DDPM'].append((self.global_step, ddpm_cur_fid))

            # ===== DDIM Samplers =====
            for sampler in self.ddim_samplers:
                if cur_step != 0 and (cur_step % sampler.sample_every) == 0:
                    # Image generation
                    if sampler.generate_image:
                        with torch.inference_mode():
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            c_batch = np.insert(np.cumsum(np.array(batches)), 0, 0)
                            for i, j in zip([True, False], ['clip', 'no_clip']):
                                if sampler.clip not in [i, 'both']:
                                    continue
                                if getattr(sampler, "fixed_noise", False):
                                    imgs = []
                                    for b in range(len(batches)):
                                        imgs.append(sampler.sample(self.diffusion_model, batch_size=None, clip=i,
                                                                   noise=sampler.noise[c_batch[b]:c_batch[b+1]]))
                                else:
                                    imgs = list(map(lambda n: self.diffusion_model.sample(
                                        batch_size=n,
                                        sparse_input=x_in_for_model[:n],
                                        perceiver_input=None,
                                        mask=cond_mask_for_model[:n],
                                        clip=i
                                    ), batches))
                                imgs = torch.cat(imgs, dim=0)
                                save_image(imgs, nrow=self.nrow,
                                           fp=os.path.join(sampler.save_path, j, f'sample_{cur_step}.png'))
                                if self.writer is not None:
                                    self.writer.add_images(f'{sampler.sampler_name} sampling result ({j})', imgs, cur_step)

                    # FID evaluation
                    if sampler.calculate_fid:
                        sample_ = lambda batch_size, clip=True, min1to1=False: sampler.sample(
                            self.diffusion_model,
                            batch_size=batch_size,
                            clip=clip,
                            min1to1=min1to1,
                            sparse_input=torch.zeros(batch_size, self.diffusion_model.channel, self.image_size, self.image_size, device=self.device),
                            perceiver_input=None,
                            mask=torch.zeros(batch_size, self.diffusion_model.channel, self.image_size, self.image_size, device=self.device)
                        )
                        ddim_cur_fid, _ = self.fid_scorer.fid_score(sample_, sampler.num_fid_sample)
                        if sampler.best_fid[0] > ddim_cur_fid:
                            sampler.best_fid[0] = ddim_cur_fid
                            if sampler.save:
                                self.save(f'best_fid_{sampler.sampler_name}')
                        if sampler.sampler_name == self.tqdm_sampler_name:
                            cur_fid = ddim_cur_fid
                        if self.writer is not None:
                            self.writer.add_scalars('FID', {sampler.sampler_name: ddim_cur_fid}, cur_step)
                        self.fid_score_log[sampler.sampler_name].append((self.global_step, ddim_cur_fid))

            self.global_step += 1

        print(colored('Training Finished!', 'yellow'))
        if self.writer is not None:
            self.writer.close()

    def save(self, name):
        data = {
            'global_step': self.global_step,
            'model': self.diffusion_model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'fid_logger': self.fid_score_log,
            'tensorboard': self.tensorboard_name
        }
        for sampler in self.ddim_samplers:
            data[sampler.sampler_name] = sampler.state_dict()
        torch.save(data, os.path.join(self.result_folder, f'model_{name}.pt'))

    def load(self, path, tensorboard_path=None, no_prev_ddim_setting=False):
        if not os.path.exists(path):
            print(make_notification('ERROR', color='red', boundary='*'))
            print(colored('No saved checkpoint is detected. Please check you gave existing path!', 'red'))
            exit()
        if tensorboard_path is not None and not os.path.exists(tensorboard_path):
            print(make_notification('ERROR', color='red', boundary='*'))
            print(colored('No tensorboard is detected. Please check you gave existing path!', 'red'))
            exit()
        print(make_notification('Loading Checkpoint', color='green'))
        data = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(data['model'])
        self.global_step = data['global_step']
        self.optimizer.load_state_dict(data['opt'])
        fid_score_log = data['fid_logger']
        if no_prev_ddim_setting:
            for key, val in self.fid_score_log.items():
                if key not in fid_score_log:
                    fid_score_log[key] = val
        else:
            for sampler in self.ddim_samplers:
                sampler.load_state_dict(data[sampler.sampler_name])
        self.fid_score_log = fid_score_log
        if tensorboard_path is not None:
            self.tensorboard_name = data['tensorboard']
        print(colored('Successfully loaded checkpoint!\n', 'green'))
