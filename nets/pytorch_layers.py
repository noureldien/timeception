# coding=utf-8


import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch

from core import torch_utils

class ConvOverTimeLayer(nn.Module):
    def __init__(self):
        super(ConvOverTimeLayer, self).__init__()

        self.conv_pointwises = [nn.Conv2d(10, 1, kernel_size=1).cuda() for _ in range(1024)]

    def forward(self, input):
        # input is of shape (None, 20, 512, 7, 7)

        input_shape = torch_utils.get_shape(input)
        n_spatial_maps = input_shape[2]

        t_conv_list = []
        for i in range(n_spatial_maps):
            t = input[:, :, i]  # (None, 20, 7, 7)
            t_conv = self.conv_pointwises[i](t)  # (None, 20, 7, 7)
            t_conv = t_conv.unsqueeze(2)  # (None, 20, 1, 7, 7)
            t_conv_list.append(t_conv)

        tensor = torch.cat(t_conv_list, 2)  # (None, 20, 512, 7, 7)
        return tensor

class DepthwiseConvOverTimeLayer(nn.Module):
    def __init__(self):
        super(DepthwiseConvOverTimeLayer, self).__init__()

        self.conv_depthwise = nn.Conv2d(512, 512 * 10, kernel_size=1, groups=1024)
        self.conv_pointwises = [nn.Conv2d(10, 10, kernel_size=1).cuda() for _ in range(1024)]

    def forward(self, input):
        # input is of shape (None, 20, 512, 7, 7)
        input_shape = torch_utils.get_shape(input)
        n_spatial_maps = input_shape[2]

        tensor = input.view(-1, 512, 7, 7)  # (None * 20, 512, 7, 7)
        tensor = self.conv_depthwise(tensor)  # (None * 20, 512 * 3, 7, 7)
        tensor = tensor.view(-1, 20, 512, 10, 7, 7)  # (None, 20, 512, 3, 7, 7)
        tensor = tensor.max(dim=1)[0]  # (None, 512, 3, 7, 7)

        t_conv_list = []
        for i in range(n_spatial_maps):
            t_conv = tensor[:, i]  # (None, 3, 7, 7)
            t_conv = self.conv_pointwises[i](t_conv)  # (None, 3, 7, 7)
            t_conv = t_conv.unsqueeze(2)  # (None, 3, 1, 7, 7)
            t_conv_list.append(t_conv)

        tensor = torch.cat(t_conv_list, 2)  # (None, 3, 512, 7, 7)
        return tensor
