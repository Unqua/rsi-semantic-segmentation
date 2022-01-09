import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BRRNetAtrous(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(BRRNetAtrous, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.atrous_convs = nn.ModuleList([])
        for i in range(6):
            self.atrous_convs.append(
                nn.Sequential(
                    # TODO: PAY ATTENTION TO PADDING
                    nn.Conv2d(self.in_channels if i == 0 else self.out_channels, self.out_channels, kernel_size=3,
                              padding=2 ** i, dilation=2 ** i),
                    nn.BatchNorm2d(self.out_channels)
                )
            )

    def forward(self, x):
        outputs = x = F.relu(self.atrous_convs[0](x), inplace=True)
        for block in self.atrous_convs[1:]:
            # print(x.shape)
            x = F.relu(block(x), inplace=True)

            outputs = outputs + x
        return outputs
