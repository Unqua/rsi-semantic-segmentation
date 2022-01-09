import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from models.decoders.unet import UNetDecoder
from models.encoders.unet import UNetEncoder
from models.modules.brrnet import BRRNetAtrous
from models.utils.init import initialize_weights


class PredictModule(nn.Module):
    def __init__(self, in_channels: int, encoder_channels: List[int], center_channels: int, decoder_channels: List[int],
                 num_classes: int,
                 merge_way: str, with_bn: bool):
        super(PredictModule, self).__init__()

        self.encoder = UNetEncoder(in_channels, encoder_channels, self.with_bn)
        self.atrous_conv_block = BRRNetAtrous(encoder_channels[-1], center_channels)
        self.decoder = UNetDecoder(center_channels, decoder_channels,
                                   with_bn=with_bn)
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Sequential?
        x, x_before_pools = self.encoder(x)
        # print("encoder output {}".format(x.shape))
        x = self.atrous_conv_block(x)
        # print("middle output {}".format(x.shape))
        x = self.decoder(x, x_before_pools)
        # print("decoder output {}".format(x.shape))
        x = self.output(x)
        return x


class ResidualRefinementModule(nn.Module):
    def __init__(self, in_channels: int, center_channels: int, num_classes: int):
        super(ResidualRefinementModule, self).__init__()

        self.atrous_conv_block = BRRNetAtrous(in_channels, center_channels)
        self.output = nn.Sequential(
            nn.Conv2d(center_channels, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        inp = x
        x = self.atrous_conv_block(x)  # does inp change?
        x = self.output(x)
        x = F.sigmoid(torch.add(inp, x))
        return x


class BRRNet(nn.Module):
    def __init__(self, num_classes: int):
        super(BRRNet, self).__init__()
        self.in_channels = 4
        self.encoder_channels = [64, 128, 256]
        self.predict_center_channels = 512
        self.decoder_channels = [256, 128, 64]
        self.rrm_center_channels = 64
        self.num_classes = num_classes
        # self.merge_way = 'concat'
        self.with_bn = True

        # Predict Module
        self.predict_module = PredictModule(self.in_channels, self.encoder_channels, self.predict_center_channels,
                                            self.decoder_channels, self.num_classes, self.merge_way, self.with_bn)
        # Residual Refinement Module
        self.residual_refinement_module = ResidualRefinementModule(self.num_classes, self.rrm_center_channels,
                                                                   self.num_classes)

        initialize_weights(self.predict_module)
        initialize_weights(self.residual_refinement_module)

    def forward(self, x):
        # print("Input {}".format(x.shape))
        x = self.predict_module(x)
        # print("predict module output {}".format(x.shape))
        x = self.residual_refinement_module(x)
        # print("residual module output {}".format(x.shape))
        return x
