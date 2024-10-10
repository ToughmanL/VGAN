#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	resnet.py
 *@Time	: 2023-07-19 19:03:45
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''

import torch
from torch import nn
import torch.nn.functional as F


# Define an nn.Module class for a simple residual block with equal dimensions
class ResBlock(nn.Module):
  def __init__(self, in_size:int, out_size:int, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_size, out_size, 3, stride=stride,padding=1)
    self.conv2 = nn.Conv2d(out_size, out_size, 3, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(out_size)
    self.batchnorm2 = nn.BatchNorm2d(out_size)
    if stride != 1:
      self.downsample = nn.Conv2d(in_size, out_size, 1, stride=stride)

  def forward(self, input):
    x = F.relu(self.batchnorm1(self.conv1(input)))
    x = F.relu(self.batchnorm2(self.conv2(x)))
    if hasattr(self, 'downsample'):
      input = self.downsample(input)
    x = F.relu(input + x)
    return x


# Define an nn.Module class for a simple residual block with equal dimensions
class SampleResBlock(nn.Module):
  def __init__(self, in_size:int, out_size:int):
    super().__init__()
    self.conv1 = nn.Conv2d(in_size, out_size, 3, stride = 2, padding=1)
    self.conv2 = nn.Conv2d(out_size, out_size, 3, stride = 1, padding=1)
    self.conv3 = nn.Conv2d(in_size, out_size, 3, stride = 2, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(out_size)
    self.batchnorm2 = nn.BatchNorm2d(out_size)
    self.batchnorm3 = nn.BatchNorm2d(out_size)

  def forward(self, input):
    x = F.relu(self.batchnorm1(self.conv1(input)))
    x = F.relu(self.batchnorm2(self.conv2(x)))
    input = F.relu(self.batchnorm3(self.conv3(input))) # downsample
    x = F.relu(input + x)
    return x


# Residual Neural Network precisely quantifies dysarthria severity-level based on short-duration speech segments
class ResNetStft(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 64, 7)
    self.batch_norm = nn.BatchNorm2d(64)
    self.maxpool = nn.MaxPool2d(2, 2)
    self.regu_resnet1 = ResBlock(64, 64)
    self.regu_resnet2 = ResBlock(64, 64)
    self.regu_resnet3 = ResBlock(64, 64)
    self.samp_resnet1 = SampleResBlock(64, 128)
    self.regu_resnet4 = ResBlock(128, 128)
    self.regu_resnet5 = ResBlock(128, 128)
    self.samp_resnet2 = SampleResBlock(128, 256)
    self.regu_resnet6 = ResBlock(256, 256)
    self.regu_resnet7 = ResBlock(256, 256)
    self.samp_resnet3 = SampleResBlock(256, 512)
    self.regu_resnet8 = ResBlock(512, 512)
    self.regu_resnet9 = ResBlock(512, 512)
    self.avg_pool = nn.AvgPool2d((8, 8))
    self.dropout = nn.Dropout(p=0.3)
    self.fc = nn.Linear(6144, 1)
  
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.batch_norm(x))
    x = self.maxpool(x)
    x = self.regu_resnet1(x)
    x = self.regu_resnet2(x)
    x = self.regu_resnet3(x)
    x = self.samp_resnet1(x)
    x = self.regu_resnet4(x)
    x = self.regu_resnet5(x)
    x = self.samp_resnet2(x)
    x = self.regu_resnet6(x)
    x = self.regu_resnet7(x)
    x = self.samp_resnet3(x)
    x = self.regu_resnet8(x)
    x = self.regu_resnet9(x)
    x = self.avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.dropout(x)
    x = self.fc(x)
    return x


class ResCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 5, 2)
    self.batch_norm = nn.BatchNorm2d(32)

    self.regu_resnet1 = ResBlock(32, 32)
    self.regu_resnet2 = ResBlock(32, 32)
    self.regu_resnet3 = ResBlock(32, 32)

    self.conv2 = nn.Conv2d(32, 64, 3, 2, 2)
    self.batch_norm2 = nn.BatchNorm2d(64)

    self.regu_resnet4 = ResBlock(64, 64)
    self.regu_resnet5 = ResBlock(64, 64)
    self.regu_resnet6 = ResBlock(64, 64)

    self.MHA = nn.MultiheadAttention(256, 4, 0.1, batch_first=True)

    self.batch_norm3 = nn.BatchNorm1d(256)
    self.fc1 = nn.Linear(256, 128)
    self.batch_norm4 = nn.BatchNorm1d(128)
    self.dropout = nn.Dropout(p=0.1)
    self.fc2 = nn.Linear(128, 1)
  
  def forward(self, input):
    x = input.unsqueeze(1)
    x = self.conv1(x)
    x = F.relu(self.batch_norm(x))
    x = self.regu_resnet1(x)
    x = self.regu_resnet2(x)
    x = self.regu_resnet3(x)
    x = self.conv2(x)
    x = F.relu(self.batch_norm2(x))
    x = self.regu_resnet4(x)
    x = self.regu_resnet5(x)
    x = self.regu_resnet6(x)
    x = x.view(x.size(0), x.size(1), -1)
    x, hidden = self.MHA(x, x, x)
    x = torch.mean(x, 1)
    x = F.relu(self.batch_norm3(x))
    x = self.fc1(x)
    x = F.relu(self.batch_norm4(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x


if __name__ == '__main__':
  # stft_data = torch.randn(64, 1, 570, 450)
  # RNS = ResNetStft()
  # res = RNS.forward(stft_data)
  # print(res)

  mel_log = torch.randn(64, 1, 64, 64)
  RC = ResCNN()
  res = RC.forward(mel_log)
  print(res)