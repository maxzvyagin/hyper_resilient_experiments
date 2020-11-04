import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.conv1_1 = self.__conv(2, name='conv1_1', in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv1_2 = self.__conv(2, name='conv1_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_1 = self.__conv(2, name='conv2_1', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_2 = self.__conv(2, name='conv2_2', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_1 = self.__conv(2, name='conv3_1', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_2 = self.__conv(2, name='conv3_2', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_3 = self.__conv(2, name='conv3_3', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_1 = self.__conv(2, name='conv4_1', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_2 = self.__conv(2, name='conv4_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_3 = self.__conv(2, name='conv4_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_1 = self.__conv(2, name='conv5_1', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_2 = self.__conv(2, name='conv5_2', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_3 = self.__conv(2, name='conv5_3', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.fc6 = self.__conv(2, name='fc6', in_channels=512, out_channels=4096, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.fc7 = self.__conv(2, name='fc7', in_channels=4096, out_channels=4096, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.fc_final = self.__conv(2, name='fc-final', in_channels=4096, out_channels=21, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.ct_conv1_1 = self.__conv(2, name='ct_conv1_1', in_channels=21, out_channels=42, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.ct_conv1_2 = self.__conv(2, name='ct_conv1_2', in_channels=42, out_channels=42, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.ct_conv2_1 = self.__conv(2, name='ct_conv2_1', in_channels=42, out_channels=84, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.ct_conv3_1 = self.__conv(2, name='ct_conv3_1', in_channels=84, out_channels=168, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.ct_conv4_1 = self.__conv(2, name='ct_conv4_1', in_channels=168, out_channels=336, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.ct_conv5_1 = self.__conv(2, name='ct_conv5_1', in_channels=336, out_channels=672, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.ct_fc1 = self.__conv(2, name='ct_fc1', in_channels=672, out_channels=672, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.ct_final = self.__conv(2, name='ct_final', in_channels=672, out_channels=21, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        conv1_1         = self.conv1_1(x)
        relu1_1         = F.relu(conv1_1)
        conv1_2         = self.conv1_2(relu1_1)
        relu1_2         = F.relu(conv1_2)
        pool1_pad       = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1, pool1_idx = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2_1         = self.conv2_1(pool1)
        relu2_1         = F.relu(conv2_1)
        conv2_2         = self.conv2_2(relu2_1)
        relu2_2         = F.relu(conv2_2)
        pool2_pad       = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2, pool2_idx = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv3_1         = self.conv3_1(pool2)
        relu3_1         = F.relu(conv3_1)
        conv3_2         = self.conv3_2(relu3_1)
        relu3_2         = F.relu(conv3_2)
        conv3_3         = self.conv3_3(relu3_2)
        relu3_3         = F.relu(conv3_3)
        pool3_pad       = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3, pool3_idx = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv4_1         = self.conv4_1(pool3)
        relu4_1         = F.relu(conv4_1)
        conv4_2         = self.conv4_2(relu4_1)
        relu4_2         = F.relu(conv4_2)
        conv4_3         = self.conv4_3(relu4_2)
        relu4_3         = F.relu(conv4_3)
        conv5_1         = self.conv5_1(relu4_3)
        relu5_1         = F.relu(conv5_1)
        conv5_2         = self.conv5_2(relu5_1)
        relu5_2         = F.relu(conv5_2)
        conv5_3         = self.conv5_3(relu5_2)
        relu5_3         = F.relu(conv5_3)
        fc6             = self.fc6(relu5_3)
        relu6           = F.relu(fc6)
        drop6           = F.dropout(input = relu6, p = 0.5, training = self.training, inplace = True)
        fc7             = self.fc7(drop6)
        relu7           = F.relu(fc7)
        drop7           = F.dropout(input = relu7, p = 0.5, training = self.training, inplace = True)
        fc_final        = self.fc_final(drop7)
        ct_conv1_1_pad  = F.pad(fc_final, (33, 33, 33, 33))
        ct_conv1_1      = self.ct_conv1_1(ct_conv1_1_pad)
        ct_relu1_1      = F.relu(ct_conv1_1)
        ct_conv1_2      = self.ct_conv1_2(ct_relu1_1)
        ct_relu1_2      = F.relu(ct_conv1_2)
        ct_conv2_1      = self.ct_conv2_1(ct_relu1_2)
        ct_relu2_1      = F.relu(ct_conv2_1)
        ct_conv3_1      = self.ct_conv3_1(ct_relu2_1)
        ct_relu3_1      = F.relu(ct_conv3_1)
        ct_conv4_1      = self.ct_conv4_1(ct_relu3_1)
        ct_relu4_1      = F.relu(ct_conv4_1)
        ct_conv5_1      = self.ct_conv5_1(ct_relu4_1)
        ct_relu5_1      = F.relu(ct_conv5_1)
        ct_fc1          = self.ct_fc1(ct_relu5_1)
        ct_fc1_relu     = F.relu(ct_fc1)
        ct_final        = self.ct_final(ct_fc1_relu)
        prob            = F.softmax(ct_final)
        return prob


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

