import os
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.diffusion import GaussianDiffusion
from src.dataset import dataset_wrapper
from src.model_original import Unet
import argparse
from torch.nn.functional import mse_loss
from src.utils import FID





import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    # copied from timm/models/layers/drop.py
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    # copied from timm/models/layers/drop.py
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
        L (int): Length of data

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): The width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads))  # 2*window_size - 1, nH

        # get pair-wise relative position index for each token inside the window
        coords_w = torch.arange(self.window_size)
        relative_coords = coords_w[:, None] - coords_w[None, :]  # W, W
        relative_coords[:, :] += self.window_size - 1  # shift to start from 0
        # relative_position_index | example
        # [2, 1, 0]
        # [3, 2, 1]
        # [4, 3, 2]
        self.register_buffer("relative_position_index", relative_coords)  # (W, W): range of 0 -- 2*(W-1)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1)  # W, W, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, W, W
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, L, C = x.shape
        assert L >= self.window_size, f'input length ({L}) must be >= window size ({self.window_size})'
        assert L % self.window_size == 0, f'input length ({L}) must be divisible by window size ({self.window_size})'

        shortcut = x
        x = self.norm1(x)

        # zero-padding shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            shifted_x[:, -self.shift_size:] = 0.
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, L)  # (B, L, C)

        # reverse zero-padding shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
            x[:, :self.shift_size] = 0.  # remove invalid embs
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, L: int):
        flops = 0
        # norm1
        flops += self.dim * L
        # W-MSA/SW-MSA
        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)
        # mlp
        flops += 2 * L * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * L
        return flops


class SwinTransformerLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False, permute_reorder = True, fix_random = False
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.permute_reorder = permute_reorder
        self.perms = {i: None for i in range(depth)}
        self.fix_random = fix_random

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0, # with this now it is not swin at all
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, pos = None):
        if pos is not None:
            x = x + pos
        
        orig_x = x
        for i, blk in enumerate(self.blocks):
            # if self.fix_random:
            #     if self.perms[i] is None:
            #         perm = torch.randperm(x.size(1))
            #         self.perms[i] = perm
            #     else:
            #         perm = self.perms[i]
            # else:
            #     perm = torch.randperm(x.size(1))
                
            # x = x[:, perm]
                
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x) + orig_x
            else:
                x = blk(x) + orig_x
                
            # if self.permute_reorder:
            #     reverse_perm = torch.argsort(perm)
            #     x = x[:, reverse_perm]
                
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}, num_heads={self.num_heads}, window_size={self.window_size}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

    
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange, repeat
import ssl
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from math import pi, log
from functools import wraps
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
# Fix for torchvision dataset download issue
ssl._create_default_https_context = ssl._create_unverified_context
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===============================================================
# --- 1. The One True Perceiver IO Model Architecture ---
# ===============================================================
D = 256
# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.latest_attn = None

    def forward(self, x, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim = -1)
        self.latest_attn = attn.detach()
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

from math import log

# This helper function creates the sinusoidal embeddings
def get_sinusoidal_embeddings(n, d):
    """
    Generates sinusoidal positional embeddings.
    
    Args:
        n (int): The number of positions (num_latents).
        d (int): The embedding dimension (latent_dim).

    Returns:
        torch.Tensor: A tensor of shape (n, d) with sinusoidal embeddings.
    """
    # Ensure latent_dim is even for sin/cos pairs
    assert d % 2 == 0, "latent_dim must be an even number for sinusoidal embeddings"
    
    position = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * -(log(10000.0) / d))
    
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def add_white_noise(coords, scale=0.01):
    return coords + torch.randn_like(coords) * scale




class CascadedBlock(nn.Module):
    def __init__(self, dim, n_latents, input_dim, cross_heads, cross_dim_head, self_heads, self_dim_head, residual_dim=None):
        super().__init__()
        self.latents = nn.Parameter(get_sinusoidal_embeddings(n_latents, dim), requires_grad=False)
        self.cross_attn = PreNorm(dim, Attention(dim, input_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=input_dim)
        self.self_attn = PreNorm(dim, Attention(dim, heads=self_heads, dim_head=self_dim_head))
        self.residual_proj = nn.Linear(residual_dim, dim) if residual_dim and residual_dim != dim else None
        self.ff = PreNorm(dim, FeedForward(dim))

    def forward(self, x, context, mask=None, residual=None):
        b = context.size(0)
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        latents = self.cross_attn(latents, context=context, mask=mask) + latents
        if residual is not None:
            if self.residual_proj:
                residual = self.residual_proj(residual)
            latents = latents + residual
        latents = self.self_attn(latents) + latents
        latents = self.ff(latents) + latents
        return latents


class CascadedPerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        queries_dim,
        logits_dim = None,
        latent_dims=(512, 512, 512),
        num_latents=(256, 256, 256),
        cross_heads = 4,
        cross_dim_head = 128,
        self_heads = 8,
        self_dim_head = 128,
        decoder_ff = False,
        
    ):
        super().__init__()
        
        assert len(latent_dims) == len(num_latents), "latent_dims and num_latents must have same length"
        
    
        # self.input_proj = nn.Linear(4, 128)
        self.input_proj = nn.Sequential(
                nn.Linear(4, 128),
                nn.GELU(),
                nn.Linear(128, 128)
            )
        self.projection_matrix = nn.Parameter(torch.randn(4, 128) / np.sqrt(4)).to(DEVICE)
        # proj = torch.randn(4, 128) / np.sqrt(4)
        # self.projection_matrix = nn.Parameter(proj.detach())  # make it a leaf tenso

        # Cascaded encoder blocks
        self.encoder_blocks = nn.ModuleList()
        prev_dim = None
        for dim, n_latents in zip(latent_dims, num_latents):
            block = CascadedBlock(
                dim=dim,
                n_latents=n_latents,
                input_dim=input_dim,
                cross_heads=cross_heads,
                cross_dim_head=cross_dim_head,
                self_heads=self_heads,
                self_dim_head=self_dim_head,
                residual_dim=prev_dim
            )
            self.encoder_blocks.append(block)
            prev_dim = dim
            
        self.decoder_swin = SwinTransformerLayer(
            dim=queries_dim,
            depth=2,                  # or 4 if you want deeper decoding
            num_heads=4,
            window_size=16,           # assuming 64x64 → 4096 tokens → 256 windows of size 16
            mlp_ratio=4.0,
            drop_path=0.1,
            use_checkpoint=False
        )

        # Decoder
        final_latent_dim = latent_dims[-1]
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, final_latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=final_latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        
        self.self_attn_blocks = nn.Sequential(*[
        nn.Sequential(
            PreNorm(latent_dims[-1], Attention(latent_dims[-1], heads=self_heads, dim_head=self_dim_head)),
            PreNorm(latent_dims[-1], FeedForward(latent_dims[-1]))
        )
        for _ in range(4)  # or 3
    ])

    def forward(self, data, mask=None, queries=None):
        b = data.size(0)
        residual = None

        
        for block in self.encoder_blocks:
            residual = block(x=residual, context=data, mask=mask, residual=residual)

            
            
            
        for sa_block in self.self_attn_blocks:
            residual = sa_block[0](residual) + residual
            residual = sa_block[1](residual) + residual
        
        if  b == 1:  # Optional: only log for one sample
            latent_std = residual.std(dim=1).mean().item()
            print(f"[Latent std]: {latent_std:.4f}")
        
        if queries is None:
            return latents

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b=b)

        x = self.decoder_cross_attn(queries, context=residual)

        # Optional: skip connection to preserve input query encoding
        x = x + queries

        # Local refinement (like SCENT)
        # x = self.decoder_swin(x)

        # Final FF
        if self.decoder_ff:
            x = x + self.decoder_ff(x)

        return self.to_logits(x)
    
def prepare_model_input(images, coords, fourier_encoder_fn):
    b, c, h, w = images.shape
    pixels = rearrange(images, 'b c h w -> b (h w) c')
    batch_coords = repeat(coords, 'n d -> b n d', b=b)
    pos_embeddings = fourier_encoder_fn(batch_coords)
    input_with_pos = torch.cat((pixels, pos_embeddings), dim=-1)
    return input_with_pos, pixels, pos_embeddings
    
class GaussianFourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn((in_features, mapping_size)) * scale)

    def forward(self, coords):
        projections = coords @ self.B
        fourier_feats = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        return fourier_feats
    
def run_perceiver_reconstruction(sparse_img: torch.Tensor, mask: torch.Tensor, 
                                  model: nn.Module, fourier_encoder: nn.Module,
                                  coords: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs a full image using PerceiverIO from sparse input.

    Args:
        sparse_img: [B, 3, H, W] sparse image with 10% visible pixels
        mask:       [B, 1, H, W] mask where 1 = known, 0 = missing
        model:      Trained PerceiverIO
        fourier_encoder: Trained Fourier encoder
        coords:     [H*W, 2] coordinate grid for image (use coords_32x32)
    
    Returns:
        reconstruction: [B, 3, H, W] full reconstructed image
    """
    B, C, H, W = sparse_img.shape

    # Prepare input with pos embeddings
    input_data, _, _ = prepare_model_input(sparse_img, coords, fourier_encoder)

    # Generate queries (standard: full grid, same for all samples)
    queries = repeat(fourier_encoder(coords), 'n d -> b n d', b=B)

    # Run model
    
    model.eval()
    fourier_encoder.eval()
    output = model(input_data, queries=queries)

    # Reshape back to image
    recon = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
    return recon


def apply_random_sparsity(imgs, sparsity=0.5):
    """
    Applies random sparsity to a batch of images.
    Args:
        imgs: [B, C, H, W]
        sparsity: float in (0, 1], fraction of visible pixels
    Returns:
        sparse_imgs: same shape as imgs, unknown pixels zeroed out
        mask: [B, 1, H, W] where 1 = known, 0 = missing
    """
    B, C, H, W = imgs.shape
    mask = (torch.rand(B, 1, H, W, device=imgs.device) < sparsity).float()
    sparse_imgs = imgs * mask
    return sparse_imgs, mask


# --- Helper Functions and Coordinate Grids ---
def create_coordinate_grid(h, w, device):
    grid = torch.stack(torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing='ij'
    ), dim=-1)
    return rearrange(grid, 'h w c -> (h w) c')


from torchvision.utils import save_image
    
if __name__ == "__main__":
    import argparse
    from torchvision.transforms import ToPILImage
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--perceiver_ckpt", type=str, default="perceiver_cifar10_sparse10.pth")
    parser.add_argument("--fourier_ckpt", type=str, default="fourier_encoder.pth")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Load dataset
    dataset = dataset_wrapper(args.dataset_path, image_size=32)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    images, *_ = next(iter(dataloader))  # Unpack image from (image, label)
    images = images[:4].to(DEVICE)

    # Apply sparsity
    sparse_images, mask = apply_random_sparsity(images, sparsity=0.1)

    # Load Perceiver and Fourier
    perceiver = CascadedPerceiverIO(
        input_dim=3 + 192,  # 3 channels + 96*2 Fourier
        queries_dim=192,
        logits_dim=3,
        latent_dims=(256, 384, 512),
        num_latents=(256, 256, 256),
        decoder_ff=True
    ).to(DEVICE)
    perceiver.load_state_dict(torch.load(args.perceiver_ckpt))
    perceiver.eval()

    fourier = GaussianFourierFeatures(in_features=2, mapping_size=96, scale=15.0).to(DEVICE)
    fourier.load_state_dict(torch.load(args.fourier_ckpt))
    fourier.eval()

    # Create coordinate grid
    coords = create_coordinate_grid(32, 32, device=DEVICE)

    # Run Perceiver reconstruction
    recon = run_perceiver_reconstruction(
        sparse_img=sparse_images,
        mask=mask,
        model=perceiver,
        fourier_encoder=fourier,
        coords=coords
    )

        # Concatenate along batch dimension for comparison
    comparison = torch.cat([images, sparse_images, recon], dim=0)  # [3B, C, H, W]

    # Save to file (will look like rows of GT | Sparse | Recon if you set nrow=batch_size)
    save_image(comparison, "reconstruction_comparison.png", nrow=images.size(0), normalize=True, value_range=(-1, 1))
    print("Saved to reconstruction_comparison.png")