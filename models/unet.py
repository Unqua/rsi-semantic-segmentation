import torch.nn as nn
import torch.nn.functional as F

from models.decoders.unet import UNetDecoder
from models.encoders.unet import UNetEncoder
from models.utils.init import initialize_weights


class UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(UNet, self).__init__()
        self.in_channels = 4
        self.encoder_channels = [64, 128, 256, 512]
        self.center_channels = 1024
        self.decoder_channels = [512, 256, 128, 64]
        self.num_classes = num_classes
        self.with_bn = True

        self.encoder = UNetEncoder(self.in_channels, self.encoder_channels, self.with_bn)
        self.center_block = nn.Sequential(
            nn.Conv2d(self.encoder_channels[-1], self.center_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.center_channels, self.center_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = UNetDecoder(self.center_channels, self.decoder_channels, self.with_bn)
        self.output = nn.Sequential(
            nn.Conv2d(self.decoder_channels[-1], self.num_classes, kernel_size=1, padding=1),
            nn.ReLU(inplace=True)
        )

        initialize_weights(self.encoder)
        initialize_weights(self.center_block)
        initialize_weights(self.decoder)
        initialize_weights(self.output)

    def forward(self, x):
        x, x_before_pools = self.encoder(x)
        x = self.center_block(x)
        x = self.decoder(x, x_before_pools)
        x = self.output(x)
        return x
