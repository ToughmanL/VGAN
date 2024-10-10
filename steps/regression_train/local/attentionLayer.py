#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 attentionLayer.py
* @Time 	:	 2023/12/17
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class AttentionLayer(nn.Module):
  def __init__(self, d_model, nhead, dropout=0.4):
    super(AttentionLayer, self).__init__()
    self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
    # self.linear1 = nn.Linear(d_model, d_model * 4)
    # self.dropout = nn.Dropout(dropout)
    # self.linear2 = nn.Linear(d_model * 4, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    # self.norm2 = nn.LayerNorm(d_model)
    # self.dropout1 = nn.Dropout(dropout)
    # self.dropout2 = nn.Dropout(dropout)
    # self.activation = F.relu

  def forward(self, src, tar):
    # src是主干，tar是加权
    # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
    src = src.transpose(0, 1) # B, T, C -> T, B, C
    tar = tar.transpose(0, 1) # B, T, C -> T, B, C
    src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0] # Q K V
    # src = src + F.dropout(src2, 0.8)*0.1
    # src = self.norm1(src)

    # src2 = self.dropout(self.activation(self.linear1(src)))
    # src2 = self.activation(self.linear2(src2))
    # src = src + self.dropout2(src2)
    # src = self.norm2(src)
    # src = src.transpose(0, 1) # T, B, C -> B, T, C
    return src

if __name__ == "__main__":
  torch.manual_seed(100)
  audio_feat = torch.randn(16, 1, 32)
  viduo_feat = torch.randn(16, 1, 32)
  AL = AttentionLayer(32, 1)
  result = AL.forward(audio_feat, viduo_feat)
  print(result)