#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	cmvn.py
 *@Time	: 2023-07-03 22:12:04
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''

import json
import torch
import numpy as np

def read_cmvn(json_path):
  with open(json_path) as fp:
    data = json.load(fp)
  for key, value in data.items():
    data[key]['mean'] = torch.unsqueeze(torch.tensor(value['mean']), dim=1)
    data[key]['var'] = torch.unsqueeze(torch.tensor(value['var']), dim=1)
  return data

def read_cmvn_np(json_path):
  with open(json_path) as fp:
    data = json.load(fp)
  for key, value in data.items():
    data[key]['mean'] = np.array(value['mean'])
    data[key]['var'] = np.array(value['var'])
  return data
