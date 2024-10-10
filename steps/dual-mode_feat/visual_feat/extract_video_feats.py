#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 extract_video_feats.py
* @Time 	:	 2023/08/14
* @Author	:	 lxk caodi
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import json
import pickle
import pandas as pd
from local.stability import Stability
from local.speed import Speed
from local.range import Range
from utils.multi_process import MultiProcess
import warnings
warnings.filterwarnings("ignore")

class VideoFeatExtract():
  def __init__(self, csv_path, raw_data_dir, new_data_dir):
    self.data_info_dict = {}
    self.file_list = []
    self._read_data_info(csv_path)
    self._get_all_files(raw_data_dir, 'pkl')
    self.new_data_dir = new_data_dir
  
  # 获取数据信息，文件名、元音起始位置
  def _read_data_info(self, csv_path):
    datainfo = pd.read_csv(csv_path)
    sel_data = datainfo[['Filename', 'vowel_st']]
    sel_data_list = sel_data.T.to_dict().values()
    self.data_info_dict = {item['Filename']:item['vowel_st'] for item in sel_data_list}

  # 获取所有需要处理的数据
  def _get_all_files(self, data_dir, suff):
    for root, dirs, files in os.walk(data_dir, followlinks=True):
      for file in files:
        file_name, file_suff = file.split('.')[0], file.split('.')[1]
        if file_suff == suff and file_name in self.data_info_dict:
          self.file_list.append(os.path.join(root, file))
  
  # 读取人脸68个点，得到一个68*2*frame_len的list
  def _read_point_pkl(self, pkl_path):
    fp = open(pkl_path, "rb")
    data = pickle.load(fp)
    return data
  
  def _write_json(self, data_dict, json_path):
    with open(json_path,"w", encoding='utf-8', newline = '\n') as f:
      f.write(json.dumps(data_dict, ensure_ascii=False, indent=1)) 
  
  def four_feat_extract(self, point_list, vowel_st):
    sta = Stability(6, 1)
    spe = Speed()
    ran = Range(4,2)
    #提取说话时3类特征
    vowel_frame_start = int(vowel_st * 30)
    stability_feat = sta.process_stability(point_list, vowel_frame_start)
    speed_feat = spe.process_speed(point_list)
    range_feat = ran.process_range(point_list) 
    #拼接4类特征
    video_feat = pd.concat([stability_feat,speed_feat, range_feat],axis=1)
    res = video_feat.to_dict('records')[0]
    return res


  # 解析视频特征
  def _feat_extract(self, pkl_path):
    pkl_file_name = os.path.basename(pkl_path)
    feat_path = os.path.join(self.new_data_dir, pkl_file_name.replace('pkl', 'json'))
    if os.path.exists(feat_path):
        return
    file_name = pkl_file_name.split('.')[0]
    point_list = self._read_point_pkl(pkl_path)
    vowel_st = self.data_info_dict[file_name]
    feats_dict = self.four_feat_extract(point_list, vowel_st)
    res = dict({file_name : feats_dict})
    self._write_json(res, feat_path)
    print("done:", file_name)

  def video_extract_inter(self, multi_num=1):
    if multi_num > 1:
      MP = MultiProcess()
      MP.multi_not_result(func=self._feat_extract, arg_list=self.file_list, process_num=multi_num)
    else:
      for file_path in self.file_list:
        self._feat_extract(file_path)
    print('All done')

if __name__ == "__main__":
  csv_path = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/dual-mode_feat/acoustic_feat/tmp/pitch_ac-acousticfeat_0803.csv' # 改成了绝对路径
  raw_data_dir = 'data/segment_data/Control/N_10001_F'
  new_data_dir = 'data/visual_json_1027'
  VFE = VideoFeatExtract(csv_path, raw_data_dir, new_data_dir)
  VFE.video_extract_inter(multi_num=1)