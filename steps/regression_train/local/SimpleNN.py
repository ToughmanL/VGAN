#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 dnn.py
* @Time 	:	 2023/02/23
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
from torch import nn
import torch.nn.functional as F
try:
  from local.CustomLayer import BatchGraphAttentionLayer
except:
  from CustomLayer import BatchGraphAttentionLayer



class LinearRegression(torch.nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.layer_1 = nn.Linear(num_features, 16)
    self.layer_out = nn.Linear(16, 1)
    self.relu = nn.ReLU()

  def forward(self, inputs):
    x = self.relu(self.layer_1(inputs))
    x = self.layer_out(x)
    return x


class DNNRegression(torch.nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.layer_1 = nn.Linear(num_features, 128)
    self.layer_2 = nn.Linear(128, 32)
    self.layer_out = nn.Linear(32, 1)
    self.relu = nn.ReLU()

  def forward(self, inputs):
    x = self.relu(self.layer_1(inputs))
    x = self.relu(self.layer_2(x))
    x = self.layer_out(x)
    return x


class DNNDropRegression(torch.nn.Module):
  def __init__(self, num_features):
    super().__init__()
    self.layer_1 = nn.Linear(num_features, 128)
    self.layer_2 = nn.Linear(128, 32)
    self.layer_out = nn.Linear(32, 1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.5)

  def forward(self, inputs):
    x = self.relu(self.layer_1(inputs))
    x = self.relu(self.dropout(x))
    x = self.relu(self.layer_2(x))
    x = self.layer_out(x)
    return x


class DNN_MFCC(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layer_1 = nn.Linear(6240, 16)
    self.layer_2 = nn.Linear(16, 32)
    self.layer_3 = nn.Linear(32, 64)
    self.layer_4 = nn.Linear(64, 128)
    self.layer_5 = nn.Linear(128, 32)
    self.layer_6 = nn.Linear(32, 1)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.25)
  
  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.relu(self.layer_1(x))
    x = self.dropout(x)
    x = self.relu(self.layer_2(x))
    x = self.dropout(x)
    x = self.relu(self.layer_3(x))
    x = self.dropout(x)
    x = self.relu(self.layer_4(x))
    x = self.dropout(x)
    x = self.relu(self.layer_5(x))
    x = self.dropout(x)
    out = self.relu(self.layer_6(x))
    return out

class CNN_MFCC(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1,padding=1)
    self.conv1_bn=nn.BatchNorm2d(16)

    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1,padding=1)
    self.conv2_bn=nn.BatchNorm2d(32)

    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1,padding=1)
    self.conv3_bn=nn.BatchNorm2d(64)

    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1,padding=1)
    self.conv4_bn=nn.BatchNorm2d(64)

    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1,padding=1)
    self.conv5_bn=nn.BatchNorm2d(128)

    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1,padding=1)
    self.conv6_bn=nn.BatchNorm2d(256)

    self.pool = nn.MaxPool2d(2, 2)
    self.dropout = nn.Dropout(p=0.2)

    # self.fc1 = nn.Linear(559872, 128)
    self.fc1 = nn.Linear(214272, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 1)

  def forward(self, input):
    # x = input.view(input.shape[0], 1, input.shape[2], input.shape[3]*input.shape[1])
    x = torch.unsqueeze(input, dim=1)
    x = self.conv1(x)
    x = F.relu(self.conv1_bn(x))
    x = self.conv2(x)
    x = F.relu(self.conv2_bn(x))
    x = self.conv3(x)
    x = F.relu(self.conv3_bn(x))
    x = self.conv4(x)
    x = F.relu(self.conv4_bn(x))
    x = self.conv5(x)
    x = F.relu(self.conv5_bn(x))
    x = self.conv6(x)
    x = F.relu(self.conv6_bn(x))
     
    x = self.pool(x)
    x = self.dropout(x)
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.fc1(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    x = self.fc3(x)
    return x


class LSTM(torch.nn.Module):
  def __init__(self, n_feature=5, n_hidden=256, n_layers=2, dropout=0.5, bidirectional=False):
    super().__init__()
    self.lstm1 = nn.LSTM(n_feature, n_hidden, dropout=dropout, batch_first=True, bidirectional=bidirectional)
    self.lstm2 = nn.LSTM(n_hidden*2, 64, dropout=dropout, batch_first=True, bidirectional=bidirectional)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(64, 1)
  
  def forward(self, x):
    output1, _ = self.lstm1(x)
    output2, (hidden2, cell2) = self.lstm2(output1)
    out = self.dropout(self.relu(hidden2))
    out = self.fc(out[-1:, :, :].squeeze())
    return out


class DNNCMLRV(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, dropout):
    super().__init__()
    hidden_size = 256
    self.a_dens = nn.Linear(hidden_size, nfeat)
    self.o_dens = nn.Linear(hidden_size, nfeat)
    self.e_dens = nn.Linear(hidden_size, nfeat)
    self.i_dens = nn.Linear(hidden_size, nfeat)
    self.u_dens = nn.Linear(hidden_size, nfeat)
    self.v_dens = nn.Linear(hidden_size, nfeat)
    self.dense = nn.Linear(nfeat*num_nodes, nhid)
    self.dropout = nn.Dropout(dropout)
    self.out_proj = nn.Linear(nhid, 1)
    self.relu = nn.ReLU()

  def forward(self, input):
    new_input_list = []
    second_dim_size = input.size(1) # num_nodes
    input = F.dropout(input, 0.5, training=self.training)
    a_data = self.a_dens(input[:, 0, :])
    o_data = self.o_dens(input[:, 1, :])
    e_data = self.e_dens(input[:, 2, :])
    i_data = self.i_dens(input[:, 3, :])
    u_data = self.u_dens(input[:, 4, :])
    v_data = self.v_dens(input[:, 5, :])
    concatenated_tensor = torch.cat([a_data,o_data,e_data,i_data,u_data,v_data], dim=1)
    x = F.dropout(concatenated_tensor, 0.4, training=self.training)
    x = self.relu(self.dense(x))
    x = self.out_proj(x)
    return x


class GAT1NN2CMLRV(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    hidden_size = 256
    self.sharelayer = nn.Linear(hidden_size, nfeat)
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*nfeat, 64)
    self.linear3 = nn.Linear(64+out_channels, 32)
    self.layer_out = nn.Linear(32, 1)

    self.relu = nn.ReLU()
  
  def get_unpad_feat(self, feat):
    piece_data = feat[0,:,0]
    unpad_len = piece_data.shape[0]
    for value in range(1, unpad_len):
      if piece_data[-value] != 0:
        unpad_len = unpad_len-value+1
        break
    return feat[:,:unpad_len,:]

  def forward(self, input):
    new_input_list = []
    second_dim_size = input.size(1) # num_nodes
    for i in range(second_dim_size):
      node_data = input[:, i, :, :]
      node_data = self.get_unpad_feat(node_data)
      node_data = torch.mean(node_data, dim=1)
      node_data = self.sharelayer(node_data)
      new_input_list.append(node_data)
    concatenated_tensor = torch.cat(new_input_list, dim=1)
    linear_x = concatenated_tensor.view(input.shape[0], input.shape[1], new_input_list[0].shape[0])
    x = torch.cat([att(linear_x) for att in self.attentions], dim=2) # 把每一个头计算出来的结果拼接一起
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.flatten(x, 1)
    x = self.relu(self.linear1(x))

    y = torch.flatten(linear_x, 1)
    y = self.relu(self.linear2(y))
    y = F.dropout(y, self.dropout, training=self.training)

    xy = torch.cat([x, y], dim=1)
    xy = self.relu(self.linear3(xy))
    xy = F.dropout(xy, self.dropout, training=self.training)
    xy = self.layer_out(xy)
    return x


class DNNWAV2VEC(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, dropout):
    super().__init__()
    hidden_size = 768
    self.linear1 = nn.Linear(hidden_size, nhid)
    self.linear2 = nn.Linear(nhid, 512)
    self.linear3 = nn.Linear(512, 64)
    self.fc = nn.Linear(64, 1)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()

  def forward(self, input):
    input = F.dropout(input, 0.3, training=self.training)
    x = self.relu(self.linear1(input))
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear2(x))
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear3(x))
    x = self.fc(x)
    return x


class DNN(torch.nn.Module):
  def __init__(self, nfeat, nhid, dropout):
    super().__init__()
    self.linear1 = nn.Linear(nfeat, nhid*2)
    self.linear2 = nn.Linear(nhid*2, nhid)
    self.linear3 = nn.Linear(nhid, nhid//2)
    self.fc = nn.Linear(nhid//2, 1)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()

  def forward(self, input):
    input = input.view(input.size(0), -1) # for mfcc cqcc
    input = F.dropout(input, 0.3, training=self.training)
    x = self.relu(self.linear1(input))
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear2(x))
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear3(x))
    x = self.fc(x)
    return x


if __name__ == '__main__':
  data = torch.randn(128, 6, 256)
  DCMLRV = DNNCMLRV(6, 128, 64, 0.2)
  res = DCMLRV.forward(data)
  print(res)
