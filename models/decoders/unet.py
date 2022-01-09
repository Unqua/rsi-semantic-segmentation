import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class UNetDecoderBlock(nn.Module):
    # up-conv->concat->conv->conv
    def __init__(self, in_channels: int, out_channels: int, with_bn: bool):
        super(UNetDecoderBlock, self).__init__()
        # align channels
        # up_conv and conv1's input are both equal to in_channels
        # pay attention to stride: output = (original-1)*stride+kernel size-padding*2
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        if with_bn:
            self.convbnrelux2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.convbnrelux2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, encoder_output):
        # print("before up_conv:{}".format(x.shape))
        x = self.up_conv(x)
        # print("after up_conv:{}".format(x.shape))

        # print("before concat: {}".format(x.shape))
        # NCHW
        x = torch.cat((encoder_output, x), dim=1)
        # print("after concat: {}".format(x.shape))
        x = self.convbnrelux2(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: List[int], with_bn: bool):
        super(UNetDecoder, self).__init__()
        self.decoder = nn.ModuleList(
            [UNetDecoderBlock(in_channels, out_channels[0], with_bn)])
        for i in range(len(out_channels) - 1):
            self.decoder.append(
                UNetDecoderBlock(out_channels[i], out_channels[i + 1], with_bn))

    def forward(self, x, x_before_pools):
        for index, block in enumerate(self.decoder):
            x = block(x, x_before_pools[-1 * (index + 1)])
        return x
