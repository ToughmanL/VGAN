#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	get_gvme_feats.py
 *@Time	: 2023-07-02 15:34:19
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''

import os
import json
import glob
import csv
import numpy as np
from utils.multi_process import MultiProcess

# numpy格式的文件不能存储在json中，因此需要转换
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def read_json(json_path):
  with open(json_path) as fp:
    data = json.load(fp)
  if 'mfcc' == json_path.split('.')[1]:
    for key, value in data.items():
      data[key]['featdata'] = np.asarray(value['featdata'])
  return data

def write_json(dict, json_path):
  with open(json_path, 'w') as fp:
    json.dump(dict, fp, cls=NumpyEncoder)

def read_files(data_dir, suff):
  feat_path_list = []
  for root, dirs, files in os.walk(data_dir, followlinks=True):
    for file in files:
      file_l = file.split('.')
      if len(file_l) < 3:
        continue
      name = file_l[0]
      if file_l[1] == suff:
        feat_path = os.path.join(root, file)
        feat_path_list.append(feat_path)
  return feat_path_list

def get_name2text(data_process_csv):
  name2text = {}
  import pandas as pd
  df = pd.read_csv(data_process_csv)
  for index in df.index:
    filename = df.at[index, 'Filename'].split('.')[0]
    count = df.at[index, 'count']
    pinyin = df.at[index, 'Syllable']
    name2text[filename+'_'+str(count)] = pinyin
  return name2text

def read_gop_score(gop_path):
  gop_dict = {}
  with open(gop_path, 'r') as fp:
    for line in fp:
      line = line.rstrip()
      line = line.replace('[ ', '')
      line = line.replace(' ]', '')
      ll = line.split(' ')
      if len(ll) == 3: # 没有辅音，仅一个元音
        gop_dict[ll[0]] = [ll[2], ll[2]]
      elif len(ll) == 5: # 一个辅音一个元音
        gop_dict[ll[0]] = [ll[2], ll[4]]
      elif len(ll) == 7: # 双元音
        gop_dict[ll[0]] = [ll[2], ll[4]]
      else: # 其他情况
        print(line)
        exit(0)
  return gop_dict

def read_gop_feat(feat_path):
  pass

def read_suff_feat(feat_path):
  suff = feat_path.split('.')[1]
  name = os.path.basename(feat_path).split('.')[0]
  if suff == 'VFER':
    fp = open(feat_path)
    lines = fp.readlines()
    result = float(lines[1].strip('\n').split('\t')[1])
    fp.close()
  elif suff == "GNE":
    fp = open(feat_path)
    lines = fp.readlines()
    result = float(lines[1].strip('\n').split('\t')[2])
    fp.close()
  elif suff == "mfcc":
    file_data = np.loadtxt(feat_path, delimiter=',', dtype=np.float32)
    result = {'featpath':feat_path, 'featdata':file_data}
  elif suff == 'egemaps':
    with open(feat_path, newline='') as f:
      reader = csv.reader(f, delimiter=';')
      data = list(reader)
    try:
      file_data = data[1][2:]
    except:
      file_data = np.array([])
      print('Data error', feat_path)
    result = {'featpath':feat_path, 'featdata':file_data}
  elif suff == "stft":
    
    file_data = np.loadtxt(feat_path, delimiter=',', dtype=np.float32)
    result = {'featpath':feat_path, 'featdata':file_data}
  return {name:result}

def get_tmp_feats(tmp_feat_dir, segment_dir, suff, multi_num=50):
  '''
    @func get_gne_vfer
    @desc 使用matlab脚本为每个音频计算GNE 和 VFER
          将这些特征txt写成json文件，如果json文件存在仅需读取
    @param {数据地址}  
    @return {} 
  '''
  feat_dict = {}
  if os.path.exists('{}/{}.json'.format(tmp_feat_dir, suff)):
    feat_dict = read_json('{}/{}.json'.format(tmp_feat_dir, suff))
  if len(feat_dict.keys()) == 0:
    feat_list = []
    feat_path_list = read_files(segment_dir, suff)
    if multi_num == 1:
      for feat_path in feat_path_list:
        name = os.path.basename(feat_path).split('.')[0]
        feat_list.append(read_suff_feat(feat_path))
    else:
      MP = MultiProcess()
      feat_list = MP.multi_with_result(func=read_suff_feat, arg_list=feat_path_list, process_num=multi_num)
    for one_feat_dict in feat_list:
      for file_name, feat_path_data in one_feat_dict.items():
        try:
          if feat_path_data['featdata'].size != 0:
            feat_dict.update(one_feat_dict)
        except:
          print(file_name)
    write_json(feat_dict, '{}/{}.json'.format(tmp_feat_dir, suff))
  return feat_dict


if __name__ == "__main__":
  segment_dir = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/data/segment_data'
  tmp_feat_dir = 'tmp/gvmeg'
  # VFER_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'VFER', 60)
  # GNE_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'GNE', 60)
  # mfcc_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'mfcc', 60)
  # egemaps_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'egemaps', 60)
  stft_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'stft', 30)
  print('feat finished')