#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 VowelGNN.py
* @Time 	:	 2023/03/21
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
from torch import nn
# from torch_geometric.nn import GATConv
import torch.nn.functional as F
try:
  from local.CustomLayer import GraphAttentionLayer, BatchGraphAttentionLayer
  from local.vresnet import VResNet
except:
  from CustomLayer import GraphAttentionLayer, BatchGraphAttentionLayer
  from vresnet import VResNet

# https://nn.labml.ai/graphs/gat/index.html
# 原始GAT，使用geometric工具，非常慢，不适用
# class GAT(torch.nn.Module):
#   def __init__(self, num_nodes, in_channels, hidden_channels, out_channels, heads, dropout_rate):
#     super().__init__()
#     self.p = dropout_rate
#     self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=dropout_rate)
#     # On the Pubmed dataset, use `heads` output heads in `conv2`.
#     self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate)
#     self.layer_out = nn.Linear(num_nodes*out_channels, 1)

#   def forward(self, x, edge_index):
#     x = F.dropout(x, p=self.p, training=self.training)
#     x = F.elu(self.conv1(x, edge_index))
#     x = F.dropout(x, p=self.p, training=self.training)
#     x = F.relu(self.conv2(x, edge_index))
#     x = torch.flatten(x)
#     x = self.layer_out(x)
#     return x

# https://www.zhihu.com/question/338051122/answer/2282566492
# GAT经过改造速度变快很多
class GATCustom(torch.nn.Module):   
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout):
    super().__init__()
    self.dropout = dropout
    self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.out_att = GraphAttentionLayer(nhid * nheads, out_channels, dropout=dropout, concat=False) # 最后一层
    self.layer_out = nn.Linear(num_nodes*out_channels, 1)
  
  def forward(self, x):         
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.cat([att(x) for att in self.attentions], dim=1) # 把每一个头计算出来的结果拼接一起
    x = F.dropout(x, self.dropout, training=self.training)
    x = F.elu(self.out_att(x))
    x = torch.flatten(x)
    x = self.layer_out(x)
    return x


# 改造的GAT进一步改造可以批处理，训练速度上来了
class BatchGATCustom(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.out_att = BatchGraphAttentionLayer(nhid * nheads, out_channels, dropout, batch_size, concat=False) # 最后一层
    self.layer_out = nn.Linear(num_nodes*out_channels, 1)
  
  def forward(self, x):         
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.cat([att(x) for att in self.attentions], dim=2) # 把每一个头计算出来的结果拼接一起
    x = F.dropout(x, self.dropout, training=self.training)
    x = F.elu(self.out_att(x))
    x = torch.flatten(x, 1)
    x = self.layer_out(x)
    return x

# 一层GAT
class GAT1(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.out_att = BatchGraphAttentionLayer(nhid * nheads, out_channels, dropout, batch_size, concat=False) # 最后一层
    self.linear = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.layer_out = nn.Linear(out_channels, 1)
    self.leakyrelu = nn.LeakyReLU()
  
  def forward(self, x):         
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.cat([att(x) for att in self.attentions], dim=2) # 把每一个头计算出来的结果拼接一起
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.flatten(x, 1)
    x = self.leakyrelu(self.linear(x))
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.layer_out(x)
    return x

# 三层GAT
class GAT3(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.att2 = BatchGraphAttentionLayer(nhid * nheads, nhid * nheads, dropout, batch_size, concat=False) # 最后一层
    self.out_att = BatchGraphAttentionLayer(nhid * nheads, out_channels, dropout, batch_size, concat=False) # 最后一层
    self.linear = nn.Linear(num_nodes*out_channels, out_channels)
    self.layer_out = nn.Linear(out_channels, 1)
    self.leakyrelu = nn.LeakyReLU()

  def forward(self, x):         
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.cat([att(x) for att in self.attentions], dim=2) # 把每一个头计算出来的结果拼接一起
    x = F.dropout(x, self.dropout, training=self.training)
    x = F.elu(self.att2(x))
    x = F.dropout(x, self.dropout, training=self.training)
    x = F.elu(self.out_att(x))
    x = F.dropout(x, self.dropout, training=self.training)
    x = torch.flatten(x, 1)
    x = self.leakyrelu(self.linear(x))
    x = F.dropout(x, self.dropout, training=self.training)
    x = self.layer_out(x)
    return x


# 一层GAT和两层DNN结合，DNN结点64
class GAT1NN2(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*nfeat, 64)
    self.linear3 = nn.Linear(64+out_channels, 32)
    self.layer_out = nn.Linear(32, 1)

  def forward(self, input):
    x = torch.cat([att(input) for att in self.attentions], dim=2) # 把每一个头计算出来的结果拼接一起
    x = F.dropout(x, 0.6, training=self.training)
    x = torch.flatten(x, 1)
    x = self.relu(self.linear1(x))
    x = F.dropout(x, 0.4, training=self.training)

    y = torch.flatten(input, 1)
    y = self.relu(self.linear2(y))

    xy = torch.cat([x, y], dim=1)
    # xy = F.dropout(xy, 0.4, training=self.training)

    xy = self.relu(self.linear3(xy))
    xy = F.dropout(xy, 0.3, training=self.training)
    xy = self.layer_out(xy)
    return xy


# 一层GAT和两层DNN结合，DNN结点64
class VGAT1NN2(torch.nn.Module):
  def __init__(self, num_nodes, input_dim, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.sharelayer = nn.Linear(input_dim, nfeat)
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*nfeat, 64)
    self.linear3 = nn.Linear(64+out_channels, 32)
    self.layer_out = nn.Linear(32, 1)

  def forward(self, input):
    new_input_list = []
    batch_size, node_size = input.size(0), input.size(1) # num_nodes
    input = input.view(batch_size, node_size, -1)
    for i in range(node_size):
      node_data = input[:, i]
      node_data = self.sharelayer(node_data)
      new_input_list.append(node_data)
    linear_x = torch.stack(new_input_list, dim=1)
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
    return xy

# 一层GAT和两层DNN结合，DNN结点64
class GAT1NN2VHUBERT(torch.nn.Module):
  def __init__(self, num_nodes, input_dim, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__() 
    self.dropout = dropout
    self.sharelayer = nn.Linear(input_dim, nfeat)
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*nfeat, nfeat)
    self.linear3 = nn.Linear(nfeat+out_channels, nfeat//2)
    self.layer_out = nn.Linear(nfeat//2, 1)

  def forward(self, input):
    new_input_list = []
    second_dim_size = input.size(1) # num_nodes
    for i in range(second_dim_size):
      node_data = input[:, i, :]
      node_data = torch.flatten(node_data, 1)
      node_data = self.sharelayer(node_data)
      new_input_list.append(node_data)
    linear_x = torch.stack(new_input_list, dim=1)
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
    return xy


class EarlyFusion(torch.nn.Module):
  def __init__(self, num_nodes, input_dim, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__() 
    self.dropout = dropout
    self.sharelayer = nn.Linear(256, 10) # 256是cmlrv的维度,将其映射到10维
    self.attentions = [BatchGraphAttentionLayer(input_dim, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*input_dim, nfeat)
    self.linear3 = nn.Linear(nfeat+out_channels, nfeat//2)
    self.layer_out = nn.Linear(nfeat//2, 1)

  def forward(self, input):
    new_input_list = []
    second_dim_size = input.size(1) # num_nodes
    batch_size = input.size(0)
    for i in range(second_dim_size):
      cmlrv_embedding = input[:, i, :256]
      expert_feats = input[:, i, 256:]
      cmlrv_embedding = torch.flatten(cmlrv_embedding, 1)
      cmlrv_data = self.sharelayer(cmlrv_embedding)
      cmlrv_data = self.relu(cmlrv_data)
      node_data = torch.cat([cmlrv_data, expert_feats], dim=1)
      new_input_list.append(node_data)
    linear_x = torch.stack(new_input_list, dim=1)
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
    return xy


class GAT1NN2RESNET(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.vresnet = VResNet(12, 18, nfeat) # in_channels, depth, out_dim
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*nfeat, 64)
    self.linear3 = nn.Linear(64+out_channels, 32)
    self.layer_out = nn.Linear(32, 1)

  def forward(self, input):
    new_input_list = []
    second_dim_size = input.size(1) # num_nodes
    for i in range(second_dim_size):
      node_data = input[:, i, :, :, :]
      node_data = self.vresnet(node_data)
      new_input_list.append(node_data)
    linear_x = torch.stack(new_input_list, dim=1)
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
    return xy


class GAT1NN2RESNETPAPI(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.vresnet = VResNet(12, 18, nfeat) # in_channels, depth, out_dim
    fusion_dim = nfeat + 20
    self.attentions = [BatchGraphAttentionLayer(fusion_dim, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*fusion_dim, 64)
    self.linear3 = nn.Linear(64+out_channels, 32)
    self.layer_out = nn.Linear(32, 1)

  def forward(self, input):
    new_input_list = []
    second_dim_size = input.size(1) # num_nodes
    batch_size = input.size(0)
    for i in range(second_dim_size):
      node_data = input[:, i, :, :, :]
      video_data = node_data[:,:-1,:,:]
      papi_data = node_data[:,-1,:20,0]
      video_data = self.vresnet(video_data)
      fusio_data = torch.cat((video_data, papi_data), dim=1)
      new_input_list.append(fusio_data)
    linear_x = torch.stack(new_input_list, dim=1)
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
    return xy


class LOOPRESNET(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.relu = nn.ReLU()
    self.vresnet = VResNet(10, 18, 32)
    self.linear1 = nn.Linear(num_nodes*nfeat, 64)
    self.linear2 = nn.Linear(64, out_channels)  
    self.layer_out = nn.Linear(out_channels, 1)
    self.relu = nn.ReLU()

  def forward(self, input):
    x = torch.flatten(input, 1)
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear1(x))
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear2(x))
    x = self.layer_out(x)
    return x


class LOOPDNNWAV2VEC(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nfeat, nhid)
    self.linear2 = nn.Linear(nhid, out_channels)  
    self.layer_out = nn.Linear(out_channels, 1)
    self.relu = nn.ReLU()

  def forward(self, input):
    x = torch.flatten(input, 1)
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear1(x))
    x = F.dropout(x, 0.3, training=self.training)
    x = self.relu(self.linear2(x))
    x = self.layer_out(x)
    return x


if __name__ == "__main__":
  torch.manual_seed(100)
  # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size = 6, 128, 64, 32, 3, 0.3, 8

  # data = torch.randn(6, 12, 80, 80)
  # batch_data = data.repeat(batch_size, 1 ,1, 1, 1)
  # GR = GAT1NN2RESNET(num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size)
  # result = GR(batch_data)
  # print(result.size())

  # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size = 6, 256, 64, 8, 3, 0.3, 8
  # data = torch.randn(6, 256)
  # batch_data = data.repeat(batch_size, 1 ,1)
  # GR = GAT1NN2(num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size)
  # result = GR(batch_data)
  # print(result.size())

  num_nodes, input_dim, nfeat, nhid, out_channels, nheads, dropout, batch_size = 6, 400, 64, 32, 32, 3, 0.3, 64
  data = torch.randn(6, 400)
  batch_data = data.repeat(batch_size, 1 ,1)
  VGN = VGAT1NN2(num_nodes, input_dim, nfeat, nhid, out_channels, nheads, dropout, batch_size)
  result = VGN(batch_data)

  # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size = 6, 64, 64, 8, 3, 0.3, 8
  # data = torch.randn(6, 13, 80, 80)
  # batch_data = data.repeat(batch_size, 1 ,1, 1, 1)
  # GR = GAT1NN2RESNETPAPI(num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size)
  # result = GR(batch_data)

  # num_nodes, input_dim, nfeat, nhid, out_channels, nheads, dropout, batch_size = 6, 30, 64, 64, 8, 3, 0.3, 8
  # data = torch.randn(6, 276)
  # batch_data = data.repeat(batch_size, 1 ,1)
  # EF = EarlyFusion(num_nodes, input_dim, nfeat, nhid, out_channels, nheads, dropout, batch_size)
  # result = EF(batch_data)

  print(result.size())