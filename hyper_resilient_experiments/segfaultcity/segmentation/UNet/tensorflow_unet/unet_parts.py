""" Parts of the U-Net model """

import tensorflow as tf
from tensorflow import keras


class DoubleConv(keras.layers.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, first=False):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        if first:
            self.double_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=mid_channels, kernel_size=3, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=out_channels, kernel_size=3, activation="relu")
            ])
        else:
            self.double_conv = keras.models.Sequential([
                keras.layers.Conv2D(filters=mid_channels, kernel_size=3, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=out_channels, kernel_size=3, activation="relu")
            ])

    def call(self, x):
        return self.double_conv(x)


class Down(keras.layers.Layer):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = keras.models.Sequential([
            keras.layers.MaxPool2D(2),
            DoubleConv(in_channels, out_channels)]
        )

    def call(self, x):
        return self.maxpool_conv(x)


class Up(keras.layers.Layer):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = keras.layers.UpSampling2D(size=2, mode='bilinear')
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = keras.layers.Conv2DTranspose(filters=in_channels//2, kernel_size=2, strides=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def call(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = tf.concat([x2, x1], axis=1)
        return self.conv(x)


class OutConv(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = keras.layers.Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        return self.conv(x)
