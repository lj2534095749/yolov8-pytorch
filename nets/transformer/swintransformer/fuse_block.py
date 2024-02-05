import torch
from torch import nn

from nets.base.yolox_darknet import BaseConv


class Focus_Down(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1, )
        return self.conv(x)


class Focus_Up(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels // 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        y = torch.zeros((x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2), device=x.device)
        channel1 = x[:, 0::4, ...]
        channel2 = x[:, 1::4, ...]
        channel3 = x[:, 2::4, ...]
        channel4 = x[:, 3::4, ...]
        y[..., ::2, ::2] = channel1
        y[..., 1::2, ::2] = channel2
        y[..., ::2, 1::2] = channel3
        y[..., 1::2, 1::2] = channel4
        return self.conv(y)
