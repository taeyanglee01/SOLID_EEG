import torch
import torch.nn as nn
from einops import rearrange
from .utils import PositionalEncoding
import torch.nn.functional as F




# ---------- Spatial AdaLN ----------
class AdaLayerNorm2dSpatial(nn.Module):
    def __init__(self, num_channels, cond_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, num_channels, affine=False)
        self.to_gb = nn.Conv2d(cond_channels, 2 * num_channels, kernel_size=1)
        # identity init: cond_map = 0 ⇒ gamma=1, beta=0
        nn.init.zeros_(self.to_gb.weight)
        with torch.no_grad():
            self.to_gb.bias[:num_channels].fill_(1.0)   # gamma bias = 1
            self.to_gb.bias[num_channels:].zero_()      # beta bias  = 0

    def forward(self, x, cond_map):          # x: (B,C,H,W), cond_map: (B,Cc,H,W)
        gb = self.to_gb(cond_map)            # (B,2C,H,W)
        gamma, beta = gb.chunk(2, dim=1)     # (B,C,H,W) each
        return gamma * self.norm(x) + beta


# ---------- Small spatial conditioner (produces a map, not a vector) ----------
class CondEncoderCNNSpatial(nn.Module):
    def __init__(self, in_ch=1, out_ch=16, down=2):
        """
        Produces a low-cost spatial feature map with 'out_ch' channels.
        We will bilinear-upsample it to match each block's (H,W).
        """
        super().__init__()
        layers = []
        ch = 32
        layers += [nn.Conv2d(in_ch, ch, 3, padding=1), nn.GroupNorm(1, ch), nn.SiLU()]
        for _ in range(down):
            layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GroupNorm(1, ch), nn.SiLU()]
        layers += [nn.Conv2d(ch, out_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B,1,H,W) -> (B,out_ch,H/2^down,W/2^down)
        return self.net(x)


# ---------- Blocks that use SPATIAL AdaLN ----------
class ResnetBlockSpatial(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, cond_channels=16, dropout=None, groups=32):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out

        self.adaln1 = AdaLayerNorm2dSpatial(dim, cond_channels)
        self.conv1  = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim is not None else None

        self.adaln2 = AdaLayerNorm2dSpatial(dim_out, cond_channels)
        self.act    = nn.SiLU()
        self.drop   = nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv2  = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)

        self.residual_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond_map=None):
        # resize cond_map to current spatial size if needed
        if cond_map is not None and (cond_map.shape[2:] != x.shape[2:]):
            cond_map = F.interpolate(cond_map, size=x.shape[2:], mode='bilinear', align_corners=False)

        h = self.adaln1(x, cond_map)
        h = self.act(h)
        h = self.conv1(h)

        if time_emb is not None:
            h = h + self.mlp(time_emb)[..., None, None]

        h = self.adaln2(h, cond_map)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        return h + self.residual_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, groups=32):
        super().__init__()
        self.scale = dim ** (-0.5)
        self.norm = nn.GroupNorm(groups, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')
        attn = torch.softmax(torch.einsum('b i c, b j c -> b i j', q, k) * self.scale, dim=-1)
        out = torch.einsum('b i j, b j c -> b i c', attn, v)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return self.to_out(out) + x


class ResnetAttentionBlockSpatial(nn.Module):
    def __init__(self, dim, dim_out=None, time_emb_dim=None, cond_channels=16, dropout=None, groups=32):
        super().__init__()
        self.resnet = ResnetBlockSpatial(dim, dim_out, time_emb_dim, cond_channels, dropout, groups)
        self.attn   = Attention(dim_out if dim_out is not None else dim, groups)

    def forward(self, x, time_emb=None, cond_map=None):
        x = self.resnet(x, time_emb, cond_map)
        return self.attn(x)



class AdaLayerNorm2d(nn.Module):
    def __init__(self, num_channels, cond_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.GroupNorm(1, num_channels, eps=eps, affine=False)
        self.linear = nn.Linear(cond_dim, num_channels * 2)

        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias[:num_channels], 1.0)  # gamma bias
        nn.init.constant_(self.linear.bias[num_channels:], 0.0)  # beta bias

    def forward(self, x, cond):
        # cond: (B, cond_dim)
        gamma_beta = self.linear(cond)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # Each: (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x_norm = self.norm(x)
        return gamma * x_norm + beta

    
class CondEncoderCNN(nn.Module):
    """
    Tiny CNN to turn x_coarse (B, in_ch, H, W) into a single cond vector (B, out_dim).
    - One GroupNorm on input (only once).
    - A stem conv, then (layers-1) residual conv blocks (no extra norms).
    - AdaptiveAvgPool2d(1) to get a vector cheaply (learned convs do the heavy lifting).
    """
    def __init__(self, in_ch=1, hidden=64, layers=4, out_dim=128):
        super().__init__()
        assert layers >= 2, "Use at least 2 layers (stem + ≥1 residual block)."
        self.in_ch = in_ch

        self.stem_norm = nn.GroupNorm(1, in_ch, eps=1e-5, affine=True)
        self.stem      = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)

        res_blocks = []
        for _ in range(layers - 1):
            res_blocks.append(nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # cheap global squeeze after learned convs
            nn.Flatten(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):  # x: (B, in_ch, H, W)
        h = self.stem(self.stem_norm(x))
        for blk in self.res_blocks:
            h = h + blk(h)  # residual
        return self.head(h)  # (B, out_dim)
    
# original
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
#         super().__init__()

#         self.dim, self.dim_out = dim, dim_out

#         dim_out = dim if dim_out is None else dim_out
#         self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=dim)
#         self.activation1 = nn.SiLU()
#         self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), padding=1)
#         self.block1 = nn.Sequential(self.norm1, self.activation1, self.conv1)

#         self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim is not None else None

#         self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=dim_out)
#         self.activation2 = nn.SiLU()
#         self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
#         self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), padding=1)
#         self.block2 = nn.Sequential(self.norm2, self.activation2, self.dropout, self.conv2)

#         self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None):
#         hidden = self.block1(x)
#         if time_emb is not None:
#             # add in timestep embedding
#             hidden = hidden + self.mlp(time_emb)[..., None, None]  # (B, dim_out, 1, 1)
#         hidden = self.block2(hidden)
#         return hidden + self.residual_conv(x)


# CNN version (non spatial)
# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out=None, time_emb_dim=None, cond_dim=None, dropout=None, groups=32):
#         super().__init__()

#         dim_out = dim if dim_out is None else dim_out
#         self.dim, self.dim_out = dim, dim_out

#         self.adaln1 = AdaLayerNorm2d(dim, cond_dim)
#         self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)

#         self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if time_emb_dim is not None else None

#         self.adaln2 = AdaLayerNorm2d(dim_out, cond_dim)
#         self.activation = nn.SiLU()
#         self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
#         self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)

#         self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None, cond=None):
#         h = self.adaln1(x, cond)
#         h = self.activation(h)
#         h = self.conv1(h)

#         if time_emb is not None:
#             h = h + self.mlp(time_emb)[..., None, None]

#         h = self.adaln2(h, cond)
#         h = self.activation(h)
#         h = self.dropout(h)
#         h = self.conv2(h)

#         return h + self.residual_conv(x)


# class Attention(nn.Module):
#     def __init__(self, dim, groups=32):
#         super().__init__()

#         self.dim, self.dim_out = dim, dim

#         self.scale = dim ** (-0.5)  # 1 / sqrt(d_k)
#         self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
#         self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1))
#         self.to_out = nn.Conv2d(dim, dim, kernel_size=(1, 1))

#     def forward(self, x):
#         b, c, h, w = x.shape
#         qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
#         # You can think (h*w) as sequence length where c is d_k in <Attention is all you need>
#         q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), qkv)

#         """
#         q, k, v shape: (batch, seq_length, d_k)  seq_length = height*width, d_k == c == dim
#         similarity shape: (batch, seq_length, seq_length)
#         attention_score shape: (batch, seq_length, seq_length)
#         attention shape: (batch, seq_length, d_k)
#         out shape: (batch, d_k, height, width)  d_k == c == dim
#         return shape: (batch, dim, height, width)
#         """

#         similarity = torch.einsum('b i c, b j c -> b i j', q, k)  # Q(K^T)
#         attention_score = torch.softmax(similarity * self.scale, dim=-1)  # softmax(Q(K^T) / sqrt(d_k))
#         attention = torch.einsum('b i j, b j c -> b i c', attention_score, v)
#         # attention(Q, K, V) = [softmax(Q(K^T) / sqrt(d_k))]V -> Scaled Dot-Product Attention
#         out = rearrange(attention, 'b (h w) c -> b c h w', h=h, w=w)
#         return self.to_out(out) + x

# original
# class ResnetAttentionBlock(nn.Module):
#     def __init__(self, dim, dim_out=None, time_emb_dim=None, dropout=None, groups=32):
#         super().__init__()

#         self.dim, self.dim_out = dim, dim_out

#         self.resnet = ResnetBlock(dim, dim_out, time_emb_dim, dropout, groups)
#         self.attention = Attention(dim_out, groups)

#     def forward(self, x, time_emb=None):
#         x = self.resnet(x, time_emb)
#         return self.attention(x)

# cnn version (Non spatial)
# class ResnetAttentionBlock(nn.Module):
#     def __init__(self, dim, dim_out=None, time_emb_dim=None, cond_dim=None, dropout=None, groups=32):
#         super().__init__()
#         self.resnet = ResnetBlock(dim, dim_out, time_emb_dim, cond_dim, dropout, groups)
#         self.attention = Attention(dim_out, groups)

#     def forward(self, x, time_emb=None, cond=None):
#         x = self.resnet(x, time_emb, cond)
#         return self.attention(x)



class downSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.dim, self.dim_out = dim_in, dim_in

        self.downsameple = nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        return self.downsameple(x)


class upSample(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.dim, self.dim_out = dim_in, dim_in

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), padding=1))

    def forward(self, x):
        return self.upsample(x)


class Unet(nn.Module):
    def __init__(self, dim, image_size, dim_multiply=(1, 2, 4, 8), channel=1, num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0, device='cuda', groups=32):
        """
        U-net for noise prediction. Code is based on Denoising Diffusion Probabilistic Models
        https://github.com/hojonathanho/diffusion
        :param dim: See below
        :param dim_multiply: len(dim_multiply) will be the depth of U-net model with at each level i, the dimension
        of channel will be dim * dim_multiply[i]. If the input image shape is [H, W, 3] then at the lowest level,
        feature map shape will be [H/(2^(len(dim_multiply)-1), W/(2^(len(dim_multiply)-1), dim*dim_multiply[-1]]
        if not considering U-net down-up path connection.
        :param image_size: input image size
        :param channel: 3
        :param num_res_blocks: # of ResnetBlock at each level. In downward path, at each level, there will be
        num_res_blocks amount of ResnetBlock module and in upward path, at each level, there will be
        (num_res_blocks+1) amount of ResnetBlock module
        :param attn_resolutions: The feature map resolution where we will apply Attention. In DDPM paper, author
        used Attention module when resolution of feature map is 16.
        :param dropout: dropout. If set to 0 then no dropout.
        :param device: either 'cuda' or 'cpu'
        :param groups: number of groups for Group normalization.
        """
        super().__init__()
        assert dim % groups == 0, 'parameter [groups] must be divisible by parameter [dim]'

        # Attributes
        self.dim = dim
        self.channel = channel
        self.time_emb_dim = 4 * self.dim
        self.num_resolutions = len(dim_multiply)
        self.device = device
        self.resolution = [int(image_size / (2 ** i)) for i in range(self.num_resolutions)]
        self.hidden_dims = [self.dim, *map(lambda x: x * self.dim, dim_multiply)]
        self.num_res_blocks = num_res_blocks

        

        # Time embedding
        positional_encoding = PositionalEncoding(self.dim)
        self.time_mlp = nn.Sequential(
            positional_encoding, nn.Linear(self.dim, self.time_emb_dim),
            nn.SiLU(), nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        
        # Conditioning vector setup
        # self.cond_dim = 128  # can adjust if you like
        # self.condition_proj = nn.Sequential(
        #     nn.Linear(channel, self.cond_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.cond_dim, self.cond_dim)
        # )
        # self.cond_encoder = CondEncoderCNN(in_ch=channel, hidden=64, layers=4, out_dim=self.cond_dim)
        

        self.cond_channels = 16
        self.cond_encoder  = CondEncoderCNNSpatial(in_ch=1, out_ch=self.cond_channels, down=2)

        # Layer definition
        self.down_path = nn.ModuleList([])
        self.up_path = nn.ModuleList([])
        concat_dim = list()

        # Downward Path layer definition
        self.init_conv = nn.Conv2d(3, self.dim, kernel_size=(3, 3), padding=1)
        concat_dim.append(self.dim)

        for level in range(self.num_resolutions):
            d_in, d_out = self.hidden_dims[level], self.hidden_dims[level + 1]
            for block in range(num_res_blocks):
                d_in_ = d_in if block == 0 else d_out
                if self.resolution[level] in attn_resolutions:
                    self.down_path.append(
                        ResnetAttentionBlockSpatial(d_in_, d_out, self.time_emb_dim,
                                                    cond_channels=self.cond_channels, dropout=dropout, groups=groups)
                    )
                else:
                    self.down_path.append(
                        ResnetBlockSpatial(d_in_, d_out, self.time_emb_dim,
                                           cond_channels=self.cond_channels, dropout=dropout, groups=groups)
                    )
                concat_dim.append(d_out)
            if level != self.num_resolutions - 1:
                self.down_path.append(downSample(d_out))
                concat_dim.append(d_out)

        # Middle
        mid_dim = self.hidden_dims[-1]
        self.middle_resnet_attention = ResnetAttentionBlockSpatial(
            mid_dim, mid_dim, self.time_emb_dim, cond_channels=self.cond_channels, dropout=dropout, groups=groups
        )
        self.middle_resnet = ResnetBlockSpatial(
            mid_dim, mid_dim, self.time_emb_dim, cond_channels=self.cond_channels, dropout=dropout, groups=groups
        )

        # Up path
        for level in reversed(range(self.num_resolutions)):
            d_out = self.hidden_dims[level + 1]
            for block in range(num_res_blocks + 1):
                d_in = self.hidden_dims[level + 2] if block == 0 and level != self.num_resolutions - 1 else d_out
                d_in = d_in + concat_dim.pop()
                if self.resolution[level] in attn_resolutions:
                    self.up_path.append(
                        ResnetAttentionBlockSpatial(d_in, d_out, self.time_emb_dim,
                                                    cond_channels=self.cond_channels, dropout=dropout, groups=groups)
                    )
                else:
                    self.up_path.append(
                        ResnetBlockSpatial(d_in, d_out, self.time_emb_dim,
                                           cond_channels=self.cond_channels, dropout=dropout, groups=groups)
                    )
            if level != 0:
                self.up_path.append(upSample(d_out))

        assert not concat_dim, 'Error in concatenation between down and up paths.'

        # Output
        final_ch = self.hidden_dims[1]
        self.final_norm = nn.GroupNorm(groups, final_ch)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(final_ch, channel, kernel_size=3, padding=1)


    # original?
    # def forward(self, x, time, x_coarse=None, sparse_input=None, mask=None):
    #     """
    #     x is the 3-channel stack you build in diffusion.py:
    #        x = cat([noised_image, sparse_input, mask], dim=1)  => (B,3,H,W)
    #     x_coarse is the 1-channel Perceiver recon (B,1,H,W), or None to disable conditioning.
    #     """
    #     # build spatial cond map once
    #     if x_coarse is not None:
    #         cond_base = self.cond_encoder(x_coarse)   # (B, cond_ch, H/4, W/4) if down=2
    #     else:
    #         # zeros map ⇒ AdaLN defaults to identity (γ=1, β=0)
    #         b, _, h, w = x.shape
    #         cond_base = torch.zeros(b, self.cond_channels, h // 4, w // 4, device=x.device)

    #     t_emb = self.time_mlp(time)

    #     # Down
    #     concat = []
    #     x = self.init_conv(x)
    #     concat.append(x)
    #     for layer in self.down_path:
    #         if isinstance(layer, (upSample, downSample)):
    #             x = layer(x)
    #         else:
    #             # resize cond to current spatial size (lazy upsample from cond_base)
    #             cond_here = F.interpolate(cond_base, size=x.shape[2:], mode='bilinear', align_corners=False)
    #             x = layer(x, t_emb, cond_map=cond_here)
    #         concat.append(x)

    #     # Middle
    #     cond_here = F.interpolate(cond_base, size=x.shape[2:], mode='bilinear', align_corners=False)
    #     x = self.middle_resnet_attention(x, t_emb, cond_map=cond_here)
    #     x = self.middle_resnet(x, t_emb, cond_map=cond_here)

    #     # Up
    #     for layer in self.up_path:
    #         if not isinstance(layer, upSample):
    #             x = torch.cat((x, concat.pop()), dim=1)
    #             cond_here = F.interpolate(cond_base, size=x.shape[2:], mode='bilinear', align_corners=False)
    #             x = layer(x, t_emb, cond_map=cond_here)
    #         else:
    #             x = layer(x)

    #     assert not concat_dim, 'Error in concatenation between downward path and upward path.'

    #     # Output layer
    #     final_ch = self.hidden_dims[1]
    #     self.final_norm = nn.GroupNorm(groups, final_ch)
    #     self.final_activation = nn.SiLU()
    #     self.final_conv = nn.Conv2d(final_ch, channel, kernel_size=(3, 3), padding=1)


    def forward(self, x, time, x_coarse=None, sparse_input=None, mask=None):
        """
        x is the 3-channel stack you build in diffusion.py:
           x = cat([noised_image, sparse_input, mask], dim=1)  => (B,3,H,W)
        x_coarse is the 1-channel Perceiver recon (B,1,H,W), or None to disable conditioning.
        """
        # build spatial cond map once
        if x_coarse is not None:
            cond_base = self.cond_encoder(x_coarse)   # (B, cond_ch, H/4, W/4) if down=2
        else:
            # zeros map ⇒ AdaLN defaults to identity (γ=1, β=0)
            b, _, h, w = x.shape
            cond_base = torch.zeros(b, self.cond_channels, h // 4, w // 4, device=x.device)

        t_emb = self.time_mlp(time)

        # Down
        concat = []
        x = self.init_conv(x)
        concat.append(x)
        for layer in self.down_path:
            if isinstance(layer, (upSample, downSample)):
                x = layer(x)
            else:
                # resize cond to current spatial size (lazy upsample from cond_base)
                cond_here = F.interpolate(cond_base, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = layer(x, t_emb, cond_map=cond_here)
            concat.append(x)

        # Middle
        cond_here = F.interpolate(cond_base, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.middle_resnet_attention(x, t_emb, cond_map=cond_here)
        x = self.middle_resnet(x, t_emb, cond_map=cond_here)

        # Up
        for layer in self.up_path:
            if not isinstance(layer, upSample):
                x = torch.cat((x, concat.pop()), dim=1)
                cond_here = F.interpolate(cond_base, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = layer(x, t_emb, cond_map=cond_here)
            else:
                x = layer(x)

        x = self.final_activation(self.final_norm(x))
        return self.final_conv(x)

    # cnn non spatial
    # def forward(self, x, time, x_coarse=None, sparse_input=None, mask=None):
    #     """
    #     return predicted noise given x_t and t
    #     """
    #     if x_coarse is not None:
    #         cond = self.cond_encoder(x_coarse)           # (B, cond_dim)
    #         # forward once in cnn layer so its attached as it goes use a 4 layer cnn, with residual conneciton and proper normalization once at the beginning and use the same thing across all the layers (other wise too expensive)
    #     else:
    #         cond = torch.zeros(x.shape[0], self.cond_dim, device=x.device)
    #     t = self.time_mlp(time)
    #     # Downward
    #     concat = list()
    #     x = self.init_conv(x)
    #     concat.append(x)
    #     for layer in self.down_path:
    #         x = layer(x, t, cond=cond) if not isinstance(layer, (upSample, downSample)) else layer(x)
    #         concat.append(x)

    #     # Middle
    #     x = self.middle_resnet_attention(x, t,cond=cond)
    #     x = self.middle_resnet(x, t,cond=cond)

    #     # Upward
    #     for layer in self.up_path:
    #         if not isinstance(layer, upSample):
    #             x = torch.cat((x, concat.pop()), dim=1)
    #         x = layer(x, t,cond=cond) if not isinstance(layer, (upSample, downSample)) else layer(x)
    #     # separate encoder/cnn encoder: to be more spatially informed, can be attention or cnn 
    #     # Final
    #     x = self.final_activation(self.final_norm(x))
    #     return self.final_conv(x)

    def print_model_structure(self):
        for i in self.down_path:
            if i.__class__.__name__ == 'downSample':
                print('-' * 20)
            if i.__class__.__name__ == "Conv2d":

                print(i.__class__.__name__)
            else:
                print(i.__class__.__name__, i.dim, i.dim_out)
        print('\n')
        print('=' * 20)
        print('\n')
        for i in self.up_path:
            if i.__class__.__name__ == 'upSample':
                print('-' * 20)
            if i.__class__.__name__ == "Conv2d":
                print(i.__class__.__name__)
            else:
                print(i.__class__.__name__, i.dim, i.dim_out)
