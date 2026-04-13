import torch.nn as nn
from collections import OrderedDict

def create_unet_block(in_channels, out_channels, name):
    """
    Creates a standard U-Net double convolution block with BatchNorm and ReLU.
    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=out_channels)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm2", nn.BatchNorm2d(num_features=out_channels)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        )
    )
