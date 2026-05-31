import torch
import torch.nn as nn
import numpy as np


class AttrEncoder(nn.Module):
    def __init__(self, embedding_dim=128, den_cond_dim=128, hidden_dim=256):
        super(AttrEncoder, self).__init__()

        # process continuous attributes
        self.fc1 = nn.Linear(6, embedding_dim)

        # process categorical attributes
        self.departure_embedding = nn.Embedding(288, hidden_dim)
        self.sid_embedding = nn.Embedding(256, hidden_dim)
        self.eid_embedding = nn.Embedding(256, hidden_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim*3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # process pool_den_cond
        self.fc3 = nn.Linear(den_cond_dim, embedding_dim)

    def forward(self, attr, pool_den_cond):
        # Continuous attributes
        continuous_attrs = attr[:, 1:7]
        # Categorical attributes
        departure, sid, eid = attr[:, 0].int(), attr[:, 7].int(), attr[:, 8].int()

        # Continuous attributes part
        out1 = self.fc1(continuous_attrs)
        # Categorical attributes part
        departure_embed = self.departure_embedding(departure)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        categorical_embed = torch.cat((departure_embed, sid_embed, eid_embed), dim=1)
        out2 = self.fc2(categorical_embed)
        # pool_den_cond part
        out3 = self.fc3(pool_den_cond)

        # Combine wide and deep embeddings
        combined_embed = out1 + out2 + out3

        return combined_embed


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32,
                        num_channels=in_channels,
                        eps=1e-6,
                        affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 temb_channels=512,
                 dropout=0.1,
                 conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5
        self.to_q = nn.Linear(dim_q, dim_q, bias=False)
        self.to_k = nn.Linear(dim_kv, dim_q, bias=False)
        self.to_v = nn.Linear(dim_kv, dim_q, bias=False)
        self.out = nn.Linear(dim_q, dim_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        _, N_c, _ = x_kv.shape
        q = self.to_q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.to_k(x_kv).reshape(B, N_c, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.to_v(x_kv).reshape(B, N_c, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, dim_cond, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = Normalize(dim)
        self.conv_in = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim_q=dim, dim_kv=dim_cond, num_heads=num_heads, dropout=dropout)

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * ff_mult, dropout=dropout)

        self.conv_out = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, den_cond):
        """
        x: [B, C, N]
        den_cond: [B, C_cond, N]
        """
        # Conv 预处理
        x = self.norm1(x)
        x = self.conv_in(x)
        x = x.permute(0, 2, 1)  # [B, N, C]
        den_cond = den_cond.permute(0, 2, 1) # [B, N, C_cond]

        # Cross-Attention
        h = self.cross_attn(self.norm2(x), den_cond)
        x = x + h

        # Feed Forward
        h = self.ff(self.norm3(x))
        x = x + h

        # 输出卷积
        x = x.permute(0, 2, 1)  # [B, C, N]
        x = self.conv_out(x)
        return x


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ch = config.model.ch
        self.out_ch =  config.model.out_ch
        self.ch_mult = tuple(config.model.ch_mult)
        self.dim_cond = config.model.cond_ch
        self.temb_ch = config.model.ch * 4
        self.dropout = config.model.dropout
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = config.model.num_res_blocks
        self.traj_length = config.data.traj_length
        self.config = config

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        self.in_channels = config.model.in_channels
        self.conv_in = torch.nn.Conv1d(self.in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # downsampling
        self.down = nn.ModuleList()
        in_ch_mult = (1, ) + self.ch_mult
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=self.dropout))
                block.append(
                    TransformerBlock(dim=block_out, 
                                     dim_cond=self.dim_cond)
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, config.model.resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)
        self.mid.attn = TransformerBlock(dim=block_in, 
                                           dim_cond=self.dim_cond)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = self.ch * self.ch_mult[i_level]
            skip_in = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=self.dropout))
                block.append(
                    TransformerBlock(dim=block_out, 
                                     dim_cond=self.dim_cond)
                )
                block_in = block_out

            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, config.model.resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        self.out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, extra_embed=None, den_cond=None):
        assert x.shape[2] == self.traj_length

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling
        hs = [self.conv_in(x)] # [batch size, ch, seq_len]
        # print(hs[-1].shape)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[2 * i_block](hs[-1], temb)
                h = self.down[i_level].block[2 * i_block + 1](h, den_cond)
                # print(i_level, i_block, h.shape)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # print(len(hs), hs[-1].shape)
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn(h, den_cond)
        h = self.mid.block_2(h, temb)
        # print(h.shape)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1): # 额外多一层对下采样结果的处理
                ht = hs.pop()
                if ht.size(-1) != h.size(-1):
                    h = torch.nn.functional.pad(h, (0, ht.size(-1) - h.size(-1)))
                h = self.up[i_level].block[2 * i_block](torch.cat([h, ht], dim=1), temb)
                h = self.up[i_level].block[2 * i_block + 1](h, den_cond)
                # print(i_level, i_block, h.shape)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Guide_UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.guide_emb = AttrEncoder(embedding_dim=config.model.ch * 4, den_cond_dim=config.model.cond_ch)
        self.unet = Model(config)

    def forward(self, x, t, attr, den_cond):
        pool_den_cond = torch.mean(den_cond, dim=-1)
        guide_emb = self.guide_emb(attr, pool_den_cond)
        pred_noise = self.unet(x, t, guide_emb, den_cond)
        return pred_noise
