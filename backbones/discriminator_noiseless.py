# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import up_or_down_sampling
from . import dense_layer
from . import layers

dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb
#%%
class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
     
        
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        self.conv1 = nn.Sequential(
                    conv2d(in_channel, out_channel, kernel_size, padding=padding),
                    )

        
        self.conv2 = nn.Sequential(
                    conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.)
                    )
        self.dense_t1= dense(t_emb_dim, out_channel)


        self.act = act
        
            
        self.skip = nn.Sequential(
                    conv2d(in_channel, out_channel, 1, padding=0, bias=False),
                    )
        
            

    def forward(self, input, t_emb):
        
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
       
        out = self.act(out)
       
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)
        
        
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)


        return out


class Discriminator_small(nn.Module):
    """A discriminator for small images without time dependency."""

    def __init__(self, nc=1, ngf=64, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.act = act
        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlockSingleInput(ngf * 2, ngf * 2, act=act)
        self.conv2 = DownConvBlockSingleInput(ngf * 2, ngf * 4, downsample=True, act=act)
        self.conv3 = DownConvBlockSingleInput(ngf * 4, ngf * 8, downsample=True, act=act)
        self.conv4 = DownConvBlockSingleInput(ngf * 8, ngf * 8, downsample=True, act=act)
        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1, init_scale=0.)
        self.end_linear = dense(ngf * 8, 1)
        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x):
        input_x = x
        h0 = self.start_conv(input_x)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        out = self.conv4(h3)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)

        return out


class DownConvBlockSingleInput(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        downsample=False,
        act=nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Sequential(
            conv2d(in_channel, out_channel, kernel_size, padding=padding),
        )

        self.conv2 = nn.Sequential(
            conv2d(out_channel, out_channel, kernel_size, padding=padding, init_scale=0.)
        )

        self.act = act

        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False),
        )

    def forward(self, input):
        out = self.act(input)
        out = self.conv1(out)

        out = self.act(out)

        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)

        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out


class Discriminator_large(nn.Module):
    """A discriminator for large images without time dependency."""

    def __init__(self, nc=1, ngf=32, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.act = act
        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlockSingleInput(ngf * 2, ngf * 4, downsample=True, act=act)
        self.conv2 = DownConvBlockSingleInput(ngf * 4, ngf * 8, downsample=True, act=act)
        self.conv3 = DownConvBlockSingleInput(ngf * 8, ngf * 8, downsample=True, act=act)
        self.conv4 = DownConvBlockSingleInput(ngf * 8, ngf * 8, downsample=True, act=act)
        self.conv5 = DownConvBlockSingleInput(ngf * 8, ngf * 8, downsample=True, act=act)
        self.conv6 = DownConvBlockSingleInput(ngf * 8, ngf * 8, downsample=True, act=act)
        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
        self.end_linear = dense(ngf * 8, 1)
        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x):
        h = self.start_conv(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        out = self.conv6(h)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)

        return out
