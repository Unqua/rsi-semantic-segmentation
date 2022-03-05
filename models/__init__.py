import segmentation_models_pytorch as smp

from configs import CFG
from .deeplabv3 import DeepLabV3ResNet18, DeepLabV3ResNet34, DeepLabV3ResNet50, DeepLabV3ResNet101
from .unet import UNet
from .brrnet import BRRNet

def build_model(num_channels, num_classes):
    if CFG.MODEL.NAME == 'deeplabv3+resnet18':
        return DeepLabV3ResNet18(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3+resnet34':
        return DeepLabV3ResNet34(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3+resnet50':
        return DeepLabV3ResNet50(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3+resnet101':
        return DeepLabV3ResNet101(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'unet':
        return UNet(num_classes)
    elif CFG.MODEL.NAME == 'brrnet':
        return BRRNet(num_classes)
    if CFG.MODEL.NAME == 'deeplabv3':
        return smp.DeepLabV3(encoder_name=CFG.MODEL.BACKBONE.NAME,
                             in_channels=num_channels,
                             classes=num_classes)
    elif CFG.MODEL.NAME == 'deeplabv3+':
        return smp.DeepLabV3Plus(encoder_name=CFG.MODEL.BACKBONE.NAME,
                                 in_channels=num_channels,
                                 classes=num_classes)
    elif CFG.MODEL.NAME == 'pspnet':
        return smp.PSPNet(encoder_name=CFG.MODEL.BACKBONE.NAME,
                          in_channels=num_channels,
                          classes=num_classes)
    elif CFG.MODEL.NAME == 'unet':
        return smp.Unet(encoder_name=CFG.MODEL.BACKBONE.NAME,
                        in_channels=num_channels,
                        classes=num_classes)
    elif CFG.MODEL.NAME == 'unet++':
        return smp.UnetPlusPlus(encoder_name=CFG.MODEL.BACKBONE.NAME,
                                in_channels=num_channels,
                                classes=num_classes)
    else:
        raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
