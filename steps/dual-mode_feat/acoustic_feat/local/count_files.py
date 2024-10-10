#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	count_files.py
 *@Time	: 2023-06-25 10:31:18
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''
import os
import numpy as np
from scipy import stats
import math
import json
import datetime

# numpy格式的文件不能存储在json中，因此需要转换
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def count_file(data_dir):
  categ_file = {}
  for root, dirs, files in os.walk(data_dir):
    for file in files:
      suff = file.split('.')[1]
      if suff in categ_file:
        categ_file[suff].append(file)
      else:
        categ_file[suff] = [file]
  
  for key, value in categ_file.items():
    print(key, len(value))

def mfcc_len(mfcc_path):
  len_list = []
  mean_stat, var_stat = np.zeros(39, dtype=float), np.zeros(39, dtype=float)
  frame_num = 0
  print('begin: ',datetime.datetime.now())

  with open(mfcc_path) as fp:
    mfcc_dict = json.load(fp)
  for key, value in mfcc_dict.items():
    feat_data = np.asarray(value['featdata'])
    try:
      len_list.append(feat_data.shape[1])
    except:
      continue
    mean_stat += np.sum(feat_data, axis=1)
    var_stat += np.sum(np.square(feat_data), axis=1)
    frame_num += feat_data.shape[1]

  len_data = np.asarray(len_list)
  print('mean:{}, median:{}, mode:{}, var:{}'.format(np.mean(len_data), np.median(len_data), stats.mode(len_data), np.var(len_data)))
  print('histogram', np.histogram(len_data,bins=50))
  cmvn_file = 'cmvn.json'
  for i in range(len(mean_stat)):
    mean_stat[i] /= frame_num
    var_stat[i] = var_stat[i] / frame_num - mean_stat[i] * mean_stat[i]
    if var_stat[i] < 1.0e-20:
      var_stat[i] = 1.0e-20
    var_stat[i] = 1.0 / math.sqrt(var_stat[i])
  res_dict = {'mean':mean_stat, 'var':var_stat, 'frame_num':frame_num}
  return res_dict

def egemaps_len(egemaps_path):
  len_list = []
  mean_stat, var_stat = np.zeros(88, dtype=float), np.zeros(88, dtype=float)
  frame_num = 0
  feat_data_list = []
  with open(egemaps_path) as fp:
    egemaps_dict = json.load(fp)
  for key, value in egemaps_dict.items():
    feat_data = np.asarray(value['featdata']).astype(np.float)
    feat_data_list.append(feat_data)
  feat_data_arr = np.array(feat_data_list)
  mean_stat = np.mean(feat_data_arr, axis=0)
  var_stat = np.std(feat_data_arr, axis=0)
  res_dict = {'mean':mean_stat, 'var':1/var_stat}
  return res_dict

def write_cmvn(res_dict, json_path):
  with open(json_path, 'w') as fp:
    json.dump(res_dict, fp, cls=NumpyEncoder)


if __name__ == '__main__':
  mfcc_path = 'tmp/gvmeg/mfcc.json'
  egemaps_path = 'tmp/gvmeg/egemaps.json'
  cmvn_path = 'tmp/gvmeg/cmvn.json'
  # count_file(data_dir)
  egemaps_dict = egemaps_len(egemaps_path)
  mfcc_dict = mfcc_len(mfcc_path)
  write_cmvn({'mfcc':mfcc_dict, 'egemaps':egemaps_dict}, cmvn_path)