#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	av_crossmodel.py
 *@Time	: 2023-11-04 16:36:04
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''
import torch
from torch import nn
import torch.nn.functional as F

class VideoCNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.conv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.conv4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(559872, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 1)

class AudioCNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(559872, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 1)

class AudioVideoCNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.aconv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.aconv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.aconv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.aconv4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=1)
    self.vconv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1,padding=1)
    self.vconv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1,padding=1)
    self.vconv3 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,padding=1)
    self.vconv4 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(559872, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 1)


if __name__ == '__main__':
  data = torch.randn(128, 1, 13, 480)
  C2M = CNN2_MFCC()
  res = C2M.forward(data)
  print(res)