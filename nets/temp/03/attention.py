# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // ratio, channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x)
        return x


class NCBAM(nn.Module):
    """基于标准化的CBAM"""
    def __init__(self, channel, ratio=8, kernel_size=7):
        super().__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.normalization = NAM(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        x = x * self.normalization(x)
        return x


class RNCBAM(nn.Module):
    """含有残差的NCBAM"""
    def __init__(self, channel, ratio=8, kernel_size=7):
        super().__init__()
        self.channelattention = ChannelAttention(channel)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.normalization = NAM(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        x = x * self.normalization(x)
        residual = residual * self.normalization(residual)
        x = residual + x
        # x = self.sigmoid(residual + x)
        return x

class EMRNCBAM(nn.Module):
    """含有残差的NCBAM"""
    def __init__(self, channel, ratio=8, kernel_size=7, factor=32):
        super().__init__()
        self.channelattention = EMA(channel, factor=factor)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.normalization = NAM(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        x = x * self.normalization(x)
        residual = residual * self.normalization(residual)
        x = residual + x
        # x = self.sigmoid(residual + x)
        return x

class EMCBAM(nn.Module):
    """含有残差的NCBAM"""
    def __init__(self, channel, ratio=8, kernel_size=7, factor=32):
        super().__init__()
        self.channelattention = ChannelAttention(channel)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.normalization = NAM(channel)
        self.emattention = EMA(channel, factor=factor)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        x = x * self.emattention(x)
        residual = residual * self.emattention(residual)
        x = residual + x
        # x = self.sigmoid(residual + x)
        return x

if __name__ == "__main__":
    model = NCBAM(512)
    print(model)
    inputs = torch.ones(2, 512, 26, 26)
    outputs = model(inputs)
    print(outputs.size())
