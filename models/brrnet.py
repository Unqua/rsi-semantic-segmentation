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
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.center_channels = center_channels
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.merge_way = merge_way
        self.with_bn = with_bn

        self.encoder = UNetEncoder(self.in_channels, self.encoder_channels, with_bn=self.with_bn)
        self.atrous_conv_block = BRRNetAtrous(self.encoder_channels[-1], self.center_channels)
        self.decoder = UNetDecoder(self.center_channels, self.decoder_channels, merge_way=self.merge_way,
                                   with_bn=self.with_bn)
        self.output = nn.Conv2d(self.decoder_channels[-1], self.num_classes, kernel_size=1)

    def forward(self, x):
        # Sequential?
        x, x_before_pools = self.encoder(x)
        # print("encoder output {}".format(x.shape))
        x = self.atrous_conv_block(x)
        # print("middle output {}".format(x.shape))
        x = self.decoder(x,x_before_pools)
        # print("decoder output {}".format(x.shape))
        x = F.sigmoid(self.output(x))
        return x


class ResidualRefinementModule(nn.Module):
    def __init__(self, in_channels: int, center_channels: int, num_classes: int):
        super(ResidualRefinementModule, self).__init__()
        self.in_channels = in_channels
        self.center_channels = center_channels
        self.num_classes = num_classes

        self.atrous_conv_block = BRRNetAtrous(in_channels, center_channels)
        self.output = nn.Conv2d(center_channels, num_classes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        inp = x
        x = self.atrous_conv_block(x)  # does inp change?
        x = self.output(x)
        x = F.relu(self.bn(x), inplace=True)
        print()
        x = F.sigmoid(torch.add(inp, x))
        print()
        return x


class BRRNet(nn.Module):
    def __init__(self, num_classes: int):
        # TODO: You really should to refactor these params' names.
        super(BRRNet, self).__init__()
        self.in_channels = 4
        self.encoder_channels = [64, 128, 256]
        self.predict_center_channels = 512
        self.decoder_channels = [256, 128, 64]
        self.rrm_center_channels = 64
        self.num_classes = num_classes
        self.merge_way = 'concat'
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
