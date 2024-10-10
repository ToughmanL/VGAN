#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	get_base_feats.py
 *@Time	: 2023-06-11 18:58:08
 *@Author	: lxk
 *@Version	: 1.0
 *@Contact	: xk.liu@siat.ac.cn
 *@License	: (C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''

import os

# def read_data(data_dir, suff):
#   feat_dict = {}
#   for root, dirs, files in os.walk(data_dir, followlinks=True):
#     for file in files:
#       if len(file.split('.')) < 2:
#         continue
#       if suff == file.split('.')[1]:
#         feat_path = os.path.join(root, file)
#         file_name = file.split('.')[0]
#         file_data = None
#         if suff == 'mfcc':
#           file_data = np.loadtxt(feat_path, delimiter=',', dtype=np.float32)
#         elif suff == 'egemaps':
#           file_data = np.genfromtxt(feat_path, delimiter=',', dtype=np.float32)
#         tmp_dict = {'featpath':feat_path}
#         tmp_dict['featdata':file_data]
#         feat_dict[file_name] = tmp_dict
#   return feat_dict

# def get_mfcc(feat_dir):
#   mfcc_dict = read_data(feat_dir, 'mfcc')
#   return mfcc_dict

# def get_egemaps(feat_dir):
#   egemaps_dict = read_data(feat_dir, 'egemaps')
#   return egemaps_dict

def get_gop(gop_path):
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

def fix_text():
  import pandas as pd
  import warnings
  warnings.filterwarnings('ignore')
  pd.set_option('display.max_rows',30)
  pd.set_option('display.max_columns',80)

  csv_path = '../tmp/pitch_ac-acousticfeat_0625.csv'
  data_process_csv = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/20230617.csv'
  name2text = get_name2text(data_process_csv)
  df = pd.read_csv(csv_path,index_col=0)
  df = df.reset_index()
  for index in df.index:
    text = name2text[df.at[index, 'Filename']]
    df.at[index,'TEXT']= text
  df.to_csv('new.csv',index=False)


if __name__ == '__main__':
  # fix_text() # 将csvfeat里面的text从错误的文件名改为正确的TEXT
  feat_dir = ''
