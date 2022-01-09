import torch.nn as nn
import torch.nn.functional as F
from typing import List


class UNetEncoderBlock(nn.Module):
    # conv->conv->pooling
    def __init__(self, in_channels: int, out_channels: int, with_bn: bool):
        super(UNetEncoderBlock, self, ).__init__()
        self.with_bn = with_bn

        if with_bn:
            self.convbnrelux2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.convbnrelux2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )
        self.mp = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x_before_pool = x = self.convbnrelux2(x)
        x = self.mp(x)  # does 'x_before_pool' change?
        return x, x_before_pool


class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: List[int], with_bn: bool):
        super(UNetEncoder, self).__init__()
        self.encoder = nn.ModuleList([UNetEncoderBlock(in_channels, out_channels[0], with_bn)])
        for i in range(len(out_channels) - 1):
            self.encoder.append(UNetEncoderBlock(out_channels[i], out_channels[i + 1], with_bn))

    def forward(self, x):
        before_pools = []
        for block in self.encoder:
            x, x_before_pool = block(x)
            before_pools.append(x_before_pool)
        return x, before_pools
