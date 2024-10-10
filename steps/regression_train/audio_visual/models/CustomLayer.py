#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 CustomLayer.py
* @Time 	:	 2023/03/16
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import itertools
import torch
from torch import nn
from torch.autograd import Function
from labml_helpers.module import Module
import torch.nn.functional as F
import numpy as np

class division(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, inputs):
    output_list = []
    for i in range(inputs.shape[1]):
      x_i = torch.unsqueeze(inputs[:, i], 1)
      x_i[x_i==0] = 0.0001
      x_e = x_i.repeat_interleave(90, dim=1)
      output = torch.div(inputs, x_e)
      output_list.append(output)
    output = torch.cat(output_list, 1)
    return output

class division_space(nn.Module):
  def __init__(self, in_units, units):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(in_units, units))
    self.bias = nn.Parameter(torch.randn(units,))
  def forward(self, inputs):
    output_list = []
    feat_sum = torch.mm(inputs, self.weight) + self.bias
    feat_sum[feat_sum==0] = 0.0001
    output = torch.div(inputs, feat_sum)
    return output

class GraphAttentionLayer(nn.Module):
  def __init__(self, in_features, out_features, dropout, concat=True):
    super(GraphAttentionLayer, self).__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.concat = concat
    self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
    nn.init.xavier_uniform_(self.W.data, gain=1.414)
    self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
    nn.init.xavier_uniform_(self.a.data, gain=1.414)
    self.leakyrelu = nn.LeakyReLU()

  def forward(self, h):
    Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
    a_input = self._prepare_attentional_mechanism_input(Wh) # 每一个节点和其它所有节点拼在一起
    e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # 计算每一个节点与其它节点的注意力值
    # zero_vec = -9e15*torch.ones_like(e)
    # attention = torch.where(adj > 0, e, zero_vec) # 将邻接矩阵中小于0的变成负无穷
    attention = e
    attention = F.softmax(attention, dim=1) # 计算每一个节点与其它节点的注意力分数
    attention = F.dropout(attention, self.dropout, training=self.training)
    h_prime = torch.matmul(attention, Wh) # 聚合邻居节点特征更新自己
    if self.concat:
      return F.elu(h_prime) # 激活函数
    else:
      return h_prime

  def _prepare_attentional_mechanism_input(self, Wh): # 两个节点拼在一起的全排列。
    N = Wh.size()[0]
    Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
    Wh_repeated_alternating = Wh.repeat(N, 1)
    all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
    return all_combinations_matrix.view(N, N, 2 * self.out_features) 
  
  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class BatchGraphAttentionLayer(nn.Module):
  def __init__(self, in_features, out_features, dropout, batch_size, concat=True):
    super(BatchGraphAttentionLayer, self).__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.concat = concat
    self.W = nn.Parameter(torch.empty(size=(batch_size, in_features, out_features)))
    nn.init.xavier_uniform_(self.W.data, gain=1.414)
    self.a = nn.Parameter(torch.empty(size=(batch_size, 1, 2*out_features, 1)))
    nn.init.xavier_uniform_(self.a.data, gain=1.414)
    self.leakyrelu = nn.LeakyReLU()
  
  def forward(self, h):
    Wh = torch.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
    a_input = self._prepare_attentional_mechanism_input(Wh) # 每一个节点和其它所有节点拼在一起
    a_repeated = self.a.repeat(1,Wh.shape[1],1,1)
    e = self.leakyrelu(torch.matmul(a_input, a_repeated).squeeze(3)) # 计算每一个节点与其它节点的注意力值
    attention = e
    attention = F.softmax(attention, dim=2) # 计算每一个节点与其它节点的注意力分数
    attention = F.dropout(attention, self.dropout, training=self.training)
    h_prime = torch.matmul(attention, Wh) # 聚合邻居节点特征更新自己
    if self.concat:
      return F.elu(h_prime) # 激活函数
    else:
      return h_prime

  def _prepare_attentional_mechanism_input(self, Wh): # 两个节点拼在一起的全排列。
    batch_size = Wh.size()[0]
    N = Wh.size()[1]
    Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
    Wh_repeated_alternating = Wh.repeat(1,N,1)
    all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
    return all_combinations_matrix.view(batch_size, N, N, 2 * self.out_features) 
  
  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

if __name__ == "__main__":
  torch.manual_seed(100)
  nfeat = 14
  nhid = 64
  dropout = 0.6
  data = torch.randn(6,14)
  batch_data = data.unsqueeze(0)
  batch_size = 1

  GA = GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True)
  result = GA(data)


  BGA = BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True)
  bresult = BGA(batch_data)
  print(bresult)

