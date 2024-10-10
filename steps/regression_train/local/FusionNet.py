# -*- encoding: utf-8 -*-
'''
file       :FusionNet.py
Description:
Date       :2024/09/05 15:12:42
Author     :Toughman
version    :python3.8.9
'''

import torch
from torch import nn
import torch.nn.functional as F

try:
  from local.VowelGNN import GAT1NN2
except:
  from VowelGNN import GAT1NN2


class Identity(torch.nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x



class FusionType(torch.nn.Module):
  def __init__(self, fusion_type, audio_dim, dim=1):
    super(FusionType, self).__init__()
    self.fusion_type = fusion_type
    self.dim = dim
    self.crossatt = torch.nn.MultiheadAttention(audio_dim, 2, dropout=0.03)
    # self.norm = torch.nn.LayerNorm(audio_dim)

  def forward(self, audio_data, video_data):
    if self.fusion_type == 'cat':
      output = torch.cat((audio_data, video_data), dim=self.dim)
    elif self.fusion_type == 'add':
      output = audio_data + video_data
    elif self.fusion_type == 'mul':
      output = audio_data * video_data
    elif self.fusion_type == 'vaa':
      audio_data = torch.unsqueeze(audio_data, 0)
      video_data = torch.unsqueeze(video_data, 0)
      output, _ = self.crossatt(video_data, audio_data, audio_data) # audio作为基底
      output = torch.squeeze(output, 0)
    elif self.fusion_type == 'vaa+a':
      audio_data = torch.unsqueeze(audio_data, 0)
      video_data = torch.unsqueeze(video_data, 0)
      output, _ = self.crossatt(video_data, audio_data, audio_data)
      output = torch.squeeze(output, 0)
      output = output + audio_data
      # output = self.norm(output)
    else:
      raise ValueError('Fusion type not supported')
    return output
  
  def output_dim(self):
    return self.dim


class AVFusionNet(torch.nn.Module):
  def __init__(self, audio_model, video_model, fusion_type, audio_dim=20, output_dim=32, dropout=0.3):
    super(AVFusionNet, self).__init__()
    self.audio_dim = audio_dim
    audio_model.layer_out = Identity() # 重置audio_model的输出层
    self.audio_model = audio_model
    video_model.layer_out = Identity() # 重置video_model的输出层
    self.video_model = video_model
    self.fusion = FusionType(fusion_type, output_dim)
    if fusion_type == 'cat':
      self.fc = nn.Linear(output_dim * 2, 1)
    else: # sum, mul, vaa, vaa+a
      self.fc = nn.Linear(output_dim , 1)
    # self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(dropout)
    # self.batchnorm = nn.BatchNorm1d(output_dim)

  def forward(self, input):
    audio_input, video_input = input[:, :, :self.audio_dim], input[:, :, self.audio_dim:]
    audio_data = self.audio_model(audio_input)
    video_data = self.video_model(video_input)
    audio_data = self.sigmoid(audio_data)
    video_data = self.sigmoid(video_data)
    fusion_data = self.fusion(audio_data, video_data)
    output = self.fc(fusion_data)
    return output


if __name__ == "__main__":
  input = torch.randn(128, 6, 276)
  # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size
  audio_model = GAT1NN2(6, 20, 64, 32, 3, 0.3, 128) 
  video_model = GAT1NN2(6, 256, 64, 32, 3, 0.3, 128)
  fusion_model = AVFusionNet(audio_model, video_model, 'concat', 20, 32)
  output = fusion_model(input)
  print(output.shape)
