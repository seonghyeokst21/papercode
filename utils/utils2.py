import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
from tqdm import tqdm
import torch
import logging
import argparse
import numpy as np
import os
import shutil
import time
from torch.multiprocessing import Process
import torch.distributed as dist
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn import functional as F
import torch.nn as nn
from torch import optim
import torchvision
import pywt
import matplotlib.pyplot as plt
import math

class AutoencoderKL_Gray(nn.Module):
    def __init__(self, pretrained_model_name="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        # 원래 AutoencoderKL 구조 로드
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name)

        # 3채널 → 1채널에 맞게 첫/마지막 conv 수정
        old_first = self.vae.encoder.conv_in
        self.vae.encoder.conv_in = nn.Conv2d(
            1, old_first.out_channels, kernel_size=3, stride=1, padding=1
        )

        old_last = self.vae.decoder.conv_out
        self.vae.decoder.conv_out = nn.Conv2d(
            old_last.in_channels, 1, kernel_size=3, stride=1, padding=1
        )

    def encode(self, x):
        h = self.vae.encoder(x)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # KL divergence를 위한 샘플링
        std = torch.exp(0.5 * logvar)
        z = mean + torch.randn_like(mean) * std
        return z, mean, logvar

    def decode(self, z):
        dec = self.vae.post_quant_conv(z)
        x_recon = self.vae.decoder(dec)
        return x_recon

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        recon = self.decode(z)
        return recon, mean, logvar

def load_trained_vae(checkpoint_path='vae_ct_mri_shared.pt', device='cuda'):
    vae = AutoencoderKL_Gray()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    return vae

def encode_images(vae_model, images):
    """
    범용 VAE encoder
    AutoencoderKL_Gray와 diffusers AutoencoderKL 모두 지원
    """
    # 입력이 [0, 1] 범위면 [-1, 1]로 변환
    if images.min() >= 0 and images.max() <= 1:
        images = images * 2.0 - 1.0
    
    # Encode
    encoded = vae_model.encode(images)
    
    # 반환 타입에 따라 처리
    if isinstance(encoded, tuple):
        # AutoencoderKL_Gray: (z, mean, logvar)
        z = encoded[0]
    elif hasattr(encoded, 'latent_dist'):
        # diffusers AutoencoderKL: AutoencoderKLOutput
        z = encoded.latent_dist.sample()
    else:
        # 직접 tensor 반환
        z = encoded
    
    # Stable Diffusion 스케일 팩터
    z = z * 0.18215
    
    return z
def print_gpu_memory(tag=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    print(f"[GPU Memory] {tag} | allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")


def decode_latents(vae_model, latents):
    """
    범용 VAE decoder
    """
    # 스케일 팩터 역변환
    latents = latents / 0.18215
    
    # Decode
    decoded = vae_model.decode(latents)
    
    # 반환 타입에 따라 처리
    if hasattr(decoded, 'sample'):
        # diffusers AutoencoderKL: DecoderOutput
        decoded = decoded.sample
   
    decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
    return decoded

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """간단한 시간 임베딩"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleResBlock(nn.Module):
    """간단한 Residual Block with time and z conditioning"""
    def __init__(self, in_channels, out_channels, temb_channels, zemb_channels, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        # Time embedding projection
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        # Z embedding projection
        self.zemb_proj = nn.Linear(zemb_channels, out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()
        
        self.act = nn.SiLU()
        
    def forward(self, x, temb, zemb):
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time and z embeddings
        h = h + self.temb_proj(self.act(temb))[:, :, None, None]
        h = h + self.zemb_proj(self.act(zemb))[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip_conv(x)


class SimpleAttention(nn.Module):
    """간단한 Self-Attention Block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        k = k.view(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        v = v.view(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        
        # Attention
        scale = c ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(b, c, h, w)
        
        # Project and add residual
        out = self.proj(out)
        return out + x


class SimplifiedUNet(nn.Module):
    """
    Simplified U-Net that matches NCSNPP's input/output interface
    
    Takes:
    - x: input image [B, 8, H, W]  # 8 input channels
    - time_cond: time/noise conditioning [B]
    - z: latent code [B, nz]
    
    Returns:
    - output image [B, 8, H, W]  # 8 output channels
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.config = config
        self.image_size = config.image_size
        self.nf = nf = config.num_channels_dae  # Base channels (e.g., 128)
        self.ch_mult = config.ch_mult  # Channel multipliers
        self.num_res_blocks = min(config.num_res_blocks, 2)  # Limit to 2 for simplicity
        self.attn_resolutions = config.attn_resolutions
        self.dropout = config.dropout
        self.num_resolutions = len(self.ch_mult)
        self.all_resolutions = [self.image_size // (2 ** i) for i in range(self.num_resolutions)]
        self.embedding_type = config.embedding_type.lower()
        self.z_emb_dim = config.z_emb_dim
        self.not_use_tanh = config.not_use_tanh
        
        # Time embedding
        if self.embedding_type == 'fourier':
            time_dim = nf * 2
            self.time_embed = nn.Sequential(
                nn.Linear(nf, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )
        else:  # positional
            time_dim = nf
            self.time_embed = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.SiLU(),
                nn.Linear(time_dim * 4, time_dim * 4),
            )
        
        # Z (latent) embedding - simplified version
        self.z_transform = nn.Sequential(
            nn.Linear(config.nz, self.z_emb_dim),
            nn.SiLU(),
            nn.Linear(self.z_emb_dim, self.z_emb_dim),
            nn.SiLU(),
        )
        
        # Initial convolution
        self.input_channels = 8  # 입력 채널 수
        self.conv_in = nn.Conv2d(self.input_channels, nf, 3, padding=1)
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        channels = [nf]
        in_ch = nf
        
        for i_level in range(self.num_resolutions):
            out_ch = nf * self.ch_mult[i_level]
            
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                self.down_blocks.append(
                    SimpleResBlock(in_ch, out_ch, time_dim * 4, self.z_emb_dim, self.dropout)
                )
                in_ch = out_ch
                channels.append(in_ch)
                
                # Add attention if needed
                if self.all_resolutions[i_level] in self.attn_resolutions:
                    self.down_blocks.append(SimpleAttention(in_ch))
                    channels.append(in_ch)
            
            # Downsample (except for last level)
            if i_level != self.num_resolutions - 1:
                self.down_blocks.append(
                    nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
                )
                channels.append(in_ch)
        
        # Middle block
        self.mid_block = nn.ModuleList([
            SimpleResBlock(in_ch, in_ch, time_dim * 4, self.z_emb_dim, self.dropout),
            SimpleAttention(in_ch),
            SimpleResBlock(in_ch, in_ch, time_dim * 4, self.z_emb_dim, self.dropout),
        ])
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        
        for i_level in reversed(range(self.num_resolutions)):
            out_ch = nf * self.ch_mult[i_level]
            
            # Residual blocks for this resolution (num_res_blocks + 1)
            for i_block in range(self.num_res_blocks + 1):
                skip_ch = channels.pop()
                self.up_blocks.append(
                    SimpleResBlock(in_ch + skip_ch, out_ch, time_dim * 4, self.z_emb_dim, self.dropout)
                )
                in_ch = out_ch
                
            # Add attention if needed
            if self.all_resolutions[i_level] in self.attn_resolutions:
                self.up_blocks.append(SimpleAttention(in_ch))
            
            # Upsample (except for first level)
            if i_level != 0:
                self.up_blocks.append(
                    nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
                )
        
        # Final layers
        self.final_norm = nn.GroupNorm(8, in_ch)
        self.final_conv = nn.Conv2d(in_ch, self.input_channels, 3, padding=1)  # 출력도 8채널
        self.act = nn.SiLU()
        
    def forward(self, x, time_cond, z):
        """
        Forward pass matching NCSNPP interface
        
        Args:
            x: Input image [B, 8, H, W]  # 8 input channels
            time_cond: Time/noise conditioning [B]
            z: Latent code [B, nz]
            
        Returns:
            Output image [B, 8, H, W]  # 8 output channels
        """
        
        # Get embeddings
        if self.embedding_type == 'fourier':
            # Simple Fourier features for continuous noise levels
            temb = self.time_embed(torch.log(time_cond))
        else:
            # Positional embeddings for discrete timesteps
            temb = self.time_embed(time_cond)
        
        zemb = self.z_transform(z)
        
        # Normalize input to [-1, 1] if needed
        if not self.config.centered:
            x = 2 * x - 1
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsample
        hs = [h]
        block_idx = 0
        for i_level in range(self.num_resolutions):
            # Residual blocks
            for i_block in range(self.num_res_blocks):
                if isinstance(self.down_blocks[block_idx], SimpleResBlock):
                    h = self.down_blocks[block_idx](h, temb, zemb)
                else:
                    h = self.down_blocks[block_idx](h)
                block_idx += 1
                hs.append(h)
                
                # Attention
                if self.all_resolutions[i_level] in self.attn_resolutions:
                    h = self.down_blocks[block_idx](h)
                    block_idx += 1
                    hs.append(h)
            
            # Downsample
            if i_level != self.num_resolutions - 1:
                h = self.down_blocks[block_idx](h)
                block_idx += 1
                hs.append(h)
        
        # Middle
        h = self.mid_block[0](h, temb, zemb)
        h = self.mid_block[1](h)
        h = self.mid_block[2](h, temb, zemb)
        
        # Upsample
        block_idx = 0
        for i_level in reversed(range(self.num_resolutions)):
            # Residual blocks with skip connections
            for i_block in range(self.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up_blocks[block_idx](h, temb, zemb)
                block_idx += 1
            
            # Attention
            if self.all_resolutions[i_level] in self.attn_resolutions:
                h = self.up_blocks[block_idx](h)
                block_idx += 1
            
            # Upsample
            if i_level != 0:
                h = self.up_blocks[block_idx](h)
                block_idx += 1
        
        # Final layers
        h = self.final_norm(h)
        h = self.act(h)
        h = self.final_conv(h)
        
        # Apply tanh if needed
        if not self.not_use_tanh:
            return torch.tanh(h)
        else:
            return h


# Configuration class for testing
class SimpleConfig:
    """Simple configuration matching NCSNPP's interface"""
    def __init__(self):
        # Model architecture
        self.image_size = 256
        self.num_channels = 8  # Changed from 3 to 8 input channels
        self.num_channels_dae = 64  # Reduced from 128 for simplicity
        self.ch_mult = [1, 2, 2, 2]  # Channel multipliers
        self.num_res_blocks = 2
        self.attn_resolutions = [16]  # Apply attention at 16x16
        self.dropout = 0.1
        self.resamp_with_conv = True
        
        # Conditioning
        self.conditional = True
        self.embedding_type = 'positional'  # or 'fourier'
        self.fourier_scale = 16.0
        
        # Latent conditioning
        self.nz = 128  # Latent dimension
        self.z_emb_dim = 256
        self.n_mlp = 2
        
        # Progressive features (simplified - not fully implemented)
        self.progressive = 'none'
        self.progressive_input = 'none'
        self.progressive_combine = 'sum'
        
        # Other
        self.fir = False
        self.fir_kernel = [1, 3, 3, 1]
        self.skip_rescale = True
        self.resblock_type = 'biggan'
        self.not_use_tanh = False
        self.centered = False
