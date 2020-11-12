""" Full assembly of the parts to form the complete network """

from .unet_parts import *


def make_tensorflow_unet(n_channels, n_classes, bilinear=False):
    model = keras.models.Sequential()
    # downsample
    model.add(DoubleConv(n_channels, 64, first=True))
    model.add(Down(64, 128))
    model.add(Down(128, 256))
    model.add(Down(256, 512))
    factor = 2 if bilinear else 1
    # upsample
    model.add(Down(512, 1024 // factor))
    model.add(Up(1024, 512 // factor, bilinear))
    model.add(Up(512, 256 // factor, bilinear))
    model.add(Up(256, 128 // factor, bilinear))
    model.add(Up(128, 64, bilinear))
    model.add(OutConv(64, n_classes))

    return model
