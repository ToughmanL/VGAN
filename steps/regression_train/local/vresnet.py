# -*- encoding: utf-8 -*-
'''
file       :vresnet.py
Description:
Date       :2024/08/26 16:40:07
Author     :Toughman
version    :python3.8.9
'''

import torch
from torch import nn
import torchvision


class VResNet(nn.Module):
  def __init__(self, in_channels, depth, out_dim, droprate=0.3):
    super(VResNet, self).__init__()
    self.in_channels = in_channels
    self.depth = depth
    self.out_dim = out_dim
    self.droprate = droprate

    self.base_model = getattr(torchvision.models, 'resnet{}'.format(depth))(pretrained=True)
    self.base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
    self.base_model.fc = nn.Sequential(nn.Dropout(droprate),nn.Linear(512, out_dim))
  
  def forward(self, x):
    x = self.base_model(x)
    return x


if __name__ == '__main__':
  input_data = torch.randn(64, 12, 80, 80) # batch_size, in_channels, height, width
  model = VResNet(12, 18, 64)
  output = model(input_data)
