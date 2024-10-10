#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 get_cmlrv.py
* @Time 	:	 2023/12/12
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import torch
from utils.multi_process import MultiProcess

class ComputeCmlrvFeat():
  def __init__(self, root_dir):
    self.file_path_list = []
    self.get_all_files(root_dir, 'cmlrv.pt')
  
  def get_all_files(self, root_dir, suffx):
    for root, dirs, files in os.walk(root_dir, followlinks=True):
      for file in files:
        file_suff = file.split('.', 1)[-1]
        if file_suff == suffx:
          self.file_path_list.append(os.path.join(root, file))

  def compute_cmlrv(self, file_path, flag='mean'):
    cmlrv_data = torch.load(file_path, map_location=torch.device('cpu'))
    if flag == 'mean':
      new_feat_path = file_path.replace('cmlrv.pt', 'mean_cmlrv.pt')
      feat = torch.mean(cmlrv_data, dim=0)
    elif flag == 'max':
      new_feat_path = file_path.replace('cmlrv.pt', 'max_cmlrv.pt')
      feat = torch.max(cmlrv_data, 0)[0]
    if not os.path.exists(new_feat_path):
      torch.save(feat, new_feat_path)
  
  def multi_read_video(self, multi_num):
    #处理每个视频的特征是否使用多线程
    if multi_num == 1:
      for file_path in self.file_path_list:
        self.compute_cmlrv(file_path)
    else:
      MP = MultiProcess()
      MP.multi_not_result(func = self.compute_cmlrv, arg_list = self.file_path_list, process_num = multi_num)

if __name__ == '__main__':
  root_dir = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/data/segment_data/'
  CCF = ComputeCmlrvFeat(root_dir)
  CCF.multi_read_video(50)