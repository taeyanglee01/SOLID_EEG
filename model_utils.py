import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# utilities
# ----------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# (optional) τ index embedding injection
# ----------------------------
class TemporalIndexBias(nn.Module):
    """
    x_cat: (B, 2T, H, W)
    시간 index τ=0..T-1 임베딩을 noisy/cond에 role별로 bias로 더해줌.
    """
    def __init__(self, T, emb_dim=128):
        super().__init__()
        self.T = T
        self.emb = nn.Embedding(T, emb_dim)
        self.proj_x = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 1))
        self.proj_c = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 1))

    def forward(self, x_cat):
        B, C, H, W = x_cat.shape
        assert C == 2 * self.T, f"Expected (B,2T,H,W) with T={self.T}, got C={C}"

        x = x_cat[:, :self.T]      # (B,T,H,W)
        c = x_cat[:, self.T:]      # (B,T,H,W)

        tau = torch.arange(self.T, device=x_cat.device)      # (T,)
        e = self.emb(tau)                                     # (T,emb_dim)

        bx = self.proj_x(e).squeeze(-1)[None, :, None, None]  # (1,T,1,1)
        bc = self.proj_c(e).squeeze(-1)[None, :, None, None]  # (1,T,1,1)

        return torch.cat([x + bx, c + bc], dim=1)

# ----------------------------
# MLP-Mixer blocks for 2D tokenization
# ----------------------------
class MixerBlock2D(nn.Module):
    """
    입력을 (B, N, C) 토큰 시퀀스로 보고,
    - token-mixing: N 차원 mixing (time+space 토큰)
    - channel-mixing: C 차원 mixing
    를 residual로 수행.

    주의: token-mixing은 Linear를 N에 대해 쓰기 위해 (B,C,N)로 transpose해서 처리.
    """
    def __init__(self, num_tokens, dim, token_mlp_dim, channel_mlp_dim, dropout=0.0):
        super().__init__()

        self.token_mixing = PreNorm(
            dim,
            nn.Sequential(
                # x: (B,N,C) -> (B,C,N)로 바꿔서 N에 Linear
                Rearrange('b n c -> b c n'),
                nn.Linear(num_tokens, token_mlp_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout else nn.Identity(),
                nn.Linear(token_mlp_dim, num_tokens),
                Rearrange('b c n -> b n c'),
            ),
        )

        self.channel_mixing = PreNorm(dim, FeedForward(dim, channel_mlp_dim, dropout=dropout))

    def forward(self, x):  # (B,N,C)
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

# einops 없이 쓰고 싶으면 아래 간단 Rearrange 구현
class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
    def forward(self, x):
        # only supports the two patterns we used above
        if self.pattern == 'b n c -> b c n':
            return x.transpose(1, 2).contiguous()
        if self.pattern == 'b c n -> b n c':
            return x.transpose(1, 2).contiguous()
        raise ValueError(f"Unsupported pattern: {self.pattern}")

# ----------------------------
# Pre-UNet Mixer
# ----------------------------
class PreUNetMLPMixer(nn.Module):
    """
    x_cat: (B, 2T, H, W)
      1) (optional) τ bias injection
      2) patchify -> tokens
      3) Mixer blocks (token mixing + channel mixing)
      4) unpatchify -> (B, base_dim, H, W)
    """
    def __init__(
        self,
        T,
        H,
        W,
        out_channels,          # usually base_dim
        patch_size=4,
        embed_dim=256,
        depth=4,
        token_mlp_dim=512,
        channel_mlp_dim=1024,
        dropout=0.0,
        use_tau_bias=True,
        tau_emb_dim=128,
    ):
        super().__init__()
        assert H % patch_size == 0 and W % patch_size == 0, "H,W must be divisible by patch_size"
        self.T = T
        self.H = H
        self.W = W
        self.patch = patch_size
        self.use_tau_bias = use_tau_bias

        if use_tau_bias:
            self.tau_bias = TemporalIndexBias(T=T, emb_dim=tau_emb_dim)

        # patchify:
        # (B, 2T, H, W) -> (B, N, patch_dim)
        patch_dim = (2 * T) * (patch_size * patch_size)
        self.to_embed = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.SiLU(),
        )

        nH = H // patch_size
        nW = W // patch_size
        num_tokens = nH * nW  # spatial tokens; BUT channel includes time(2T)*patch^2 so time information is inside channel

        # mixer blocks
        self.blocks = nn.ModuleList([
            MixerBlock2D(
                num_tokens=num_tokens,
                dim=embed_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
                dropout=dropout
            ) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # unembed tokens back to feature map then 1x1 conv
        self.to_patch = nn.Linear(embed_dim, patch_dim)

        # after unpatchify -> (B, 2T, H, W), compress to out_channels
        self.merge = nn.Conv2d(2*T, out_channels, kernel_size=1)

    def patchify(self, x):
        # x: (B, 2T, H, W)
        B, C, H, W = x.shape
        p = self.patch
        nH, nW = H // p, W // p

        # (B, C, nH, p, nW, p) -> (B, nH*nW, C*p*p)
        x = x.view(B, C, nH, p, nW, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, nH * nW, C * p * p)
        return x

    def unpatchify(self, tokens):
        # tokens: (B, N, C*p*p) where C=2T
        B, N, D = tokens.shape
        p = self.patch
        nH, nW = self.H // p, self.W // p
        C = 2 * self.T
        assert N == nH * nW
        assert D == C * p * p

        x = tokens.view(B, nH, nW, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, self.H, self.W)
        return x

    def forward(self, x_cat):
        # (optional) τ bias injection
        if self.use_tau_bias:
            x_cat = self.tau_bias(x_cat)  # (B,2T,H,W)

        # patchify -> embed
        tok = self.patchify(x_cat)        # (B,N,patch_dim)
        tok = self.to_embed(tok)          # (B,N,embed_dim)

        # mixer
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.final_norm(tok)

        # back to patches -> unpatchify
        patches = self.to_patch(tok)      # (B,N,patch_dim)
        x_rec = self.unpatchify(patches)  # (B,2T,H,W)

        # compress channels to base_dim
        out = self.merge(x_rec)           # (B,out_channels,H,W)
        return out

class PixelTimeLSTM(nn.Module):
    """
    h: (B, C, H, W)
    C를 (G, T)로 factorize 해서 각 (g, h, w)마다 길이 T LSTM을 돌림.
    - C == G*T 이어야 함 (혹은 in_proj로 맞춘 뒤 처리)
    """
    def __init__(self, T, hidden=64, num_layers=1, dropout=0.0,
                 use_in_proj=True, in_channels=None, group_channels=None):
        super().__init__()
        self.T = T
        self.hidden = hidden
        self.num_layers = num_layers

        self.use_in_proj = use_in_proj
        if use_in_proj:
            assert in_channels is not None and group_channels is not None
            self.in_proj  = nn.Conv2d(in_channels, group_channels * T, kernel_size=1)
            self.out_proj = nn.Conv2d(group_channels * T, in_channels, kernel_size=1)
            self.G = group_channels
        else:
            self.in_proj = None
            self.out_proj = None
            self.G = None  # runtime에 결정

        # LSTM은 feature dim=1(스칼라)로 두고 hidden으로 확장해도 되고,
        # 더 강하게 하려면 input_size>1로도 가능하지만 여기선 안정적인 스칼라 입력 버전.
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.to_scalar = nn.Linear(hidden, 1)

    def forward(self, h):
        B, C, H, W = h.shape
        T = self.T

        if self.use_in_proj:
            y = self.in_proj(h)  # (B, G*T, H, W)
            G = self.G
        else:
            assert C % T == 0, f"C={C} must be divisible by T={T}"
            y = h
            G = C // T

        # (B, G*T, H, W) -> (B, G, T, H, W)
        y = y.view(B, G, T, H, W)

        # LSTM 입력 만들기: 각 (B, G, H, W)를 하나의 batch로 보고, 길이 T 시퀀스
        # (B, G, H, W, T) -> (B*G*H*W, T, 1)
        seq = y.permute(0, 1, 3, 4, 2).contiguous().view(B*G*H*W, T, 1)

        out, _ = self.lstm(seq)           # (B*G*H*W, T, hidden)
        out = self.to_scalar(out)         # (B*G*H*W, T, 1)
        out = out.view(B, G, H, W, T).permute(0, 1, 4, 2, 3).contiguous()  # (B,G,T,H,W)

        out = out.view(B, G*T, H, W)      # (B, G*T, H, W)

        if self.use_in_proj:
            out = self.out_proj(out)      # (B, C, H, W)

        return out

class PixelTemporalAttention(nn.Module):
    """
    x: (B, T, H, W)  ->  각 (h,w)별로 길이 T self-attention
    """
    def __init__(self, T, d_model=64, n_heads=4, dropout=0.0):
        super().__init__()
        self.T = T
        self.d_model = d_model

        # scalar(1) -> d_model
        self.in_proj = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(d_model*4, d_model),
        )
        self.out_proj = nn.Linear(d_model, 1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):  # (B,T,H,W)
        B, T, H, W = x.shape
        assert T == self.T

        # (B,T,H,W) -> (BHW, T, 1)
        seq = x.permute(0, 2, 3, 1).contiguous().view(B*H*W, T, 1)
        z = self.in_proj(seq)  # (BHW,T,d)

        # attn block
        z2, _ = self.attn(self.norm1(z), self.norm1(z), self.norm1(z), need_weights=False)
        z = z + z2

        # ff block
        z = z + self.ff(self.norm2(z))

        y = self.out_proj(z)  # (BHW,T,1)
        y = y.view(B, H, W, T).permute(0, 3, 1, 2).contiguous()  # (B,T,H,W)
        return y


class PreUNetTemporalAttnMixer(nn.Module):
    """
    x_cat (B,2T,H,W) -> noisy/cond 분리 -> 각각 temporal attention -> concat -> 1x1 conv로 base_dim 압축
    """
    def __init__(self, T, out_channels, d_model=64, heads=4, dropout=0.0):
        super().__init__()
        self.T = T
        self.attn_x = PixelTemporalAttention(T, d_model=d_model, n_heads=heads, dropout=dropout)
        self.attn_c = PixelTemporalAttention(T, d_model=d_model, n_heads=heads, dropout=dropout)
        self.merge = nn.Conv2d(2*T, out_channels, kernel_size=1)

    def forward(self, x_cat):  # (B,2T,H,W)
        B, C, H, W = x_cat.shape
        T = self.T
        assert C == 2*T

        x = x_cat[:, :T]  # (B,T,H,W)
        c = x_cat[:, T:]  # (B,T,H,W)

        x = self.attn_x(x)
        c = self.attn_c(c)

        return self.merge(torch.cat([x, c], dim=1))  # (B,out_channels,H,W)