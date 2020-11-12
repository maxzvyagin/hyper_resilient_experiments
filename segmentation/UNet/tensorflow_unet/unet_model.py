""" Full assembly of the parts to form the complete network """

from .unet_parts import *


def make_tensorflow_unet(n_channels, n_classes, bilinear=False):
    model = keras.models.Sequential()
    # downsample
    x = model.add(DoubleConv(n_channels, 64, first=True))
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


class TensorFlow_UNet_Model(keras.models.Model):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(TensorFlow_UNet_Model, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def call(self, inputs):
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits