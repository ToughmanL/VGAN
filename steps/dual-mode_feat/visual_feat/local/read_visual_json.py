#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 read_visual_json.py
* @Time 	:	 2023/08/15
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import os
import json
import pandas as pd
from utils.multi_process import MultiProcess

class ReadVisualJson():
  def __init__(self, json_dir) -> None:
    self.path_list = []
    self.papi_feats = ['Person', 'tougue_dist', 'jaw_dist', 'move_degree', 'VSA', 'VAI', 'FCR', 'u-Jitter', 'u-Shimmer', 'u-HNR', 'u-gne', 'u-vfer', 'u-F1_sd', 'u-F2_sd', 'u-F3_sd', 'u-Intensity_mean', 'u-Intensity_sd', 'u-Vowel_dur', 'u-Syllable_dur', 'u-gop_con', 'u-gop_vow', 'i-Jitter', 'i-Shimmer', 'i-HNR', 'i-gne', 'i-vfer', 'i-F1_sd', 'i-F2_sd', 'i-F3_sd', 'i-Intensity_mean', 'i-Intensity_sd', 'i-Vowel_dur', 'i-Syllable_dur', 'i-gop_con', 'i-gop_vow', 'a-Jitter', 'a-Shimmer', 'a-HNR', 'a-gne', 'a-vfer', 'a-F1_sd', 'a-F2_sd', 'a-F3_sd', 'a-Intensity_mean', 'a-Intensity_sd', 'a-Vowel_dur', 'a-Syllable_dur', 'a-gop_con', 'a-gop_vow', 'e-Jitter', 'e-Shimmer', 'e-HNR', 'e-gne', 'e-vfer', 'e-F1_sd', 'e-F2_sd', 'e-F3_sd', 'e-Intensity_mean', 'e-Intensity_sd', 'e-Vowel_dur', 'e-Syllable_dur', 'e-gop_con', 'e-gop_vow', 'o-Jitter', 'o-Shimmer', 'o-HNR', 'o-gne', 'o-vfer', 'o-F1_sd', 'o-F2_sd', 'o-F3_sd', 'o-Intensity_mean', 'o-Intensity_sd', 'o-Vowel_dur', 'o-Syllable_dur', 'o-gop_con', 'o-gop_vow', 'v-Jitter', 'v-Shimmer', 'v-HNR', 'v-gne', 'v-vfer', 'v-F1_sd', 'v-F2_sd', 'v-F3_sd', 'v-Intensity_mean', 'v-Intensity_sd', 'v-Vowel_dur', 'v-Syllable_dur', 'v-gop_con', 'v-gop_vow']
    self.mfcc_egemaps = ['u_mfcc', 'u_egemaps', 'i_mfcc', 'i_egemaps', 'a_mfcc', 'a_egemaps', 'e_mfcc', 'e_egemaps', 'o_mfcc', 'o_egemaps', 'v_mfcc', 'v_egemaps']
    self.visual_feats = {'alpha_stability':None, 'beta_stability':None, 'dist_stability':None, 'alpha_speed':None, 'beta_speed':None, 'inner_speed':None, 'inner_dist_min':None, 'inner_dist_max':None, 'w_min':None, 'w_max':None}
    self.labels = ['Frenchay','fanshe','huxi','chun','he','ruane','hou','she','yanyu']
    self.visual_feat_dict = {}
    self._get_all_josn(json_dir)
    self._read_all_json()

  def _json2dict(self, json_path):
    with open(json_path, 'r') as json_file:
      json_data = json_file.read()
    data_dict = json.loads(json_data)
    return data_dict

  def _get_all_josn(self, json_dir):
    for root, dirs, files in os.walk(json_dir, followlinks=True):
      for file in files:
        if file.split('.')[1] == 'json':
          self.path_list.append(os.path.join(root, file))
  
  def _read_all_json(self):
    self.visual_feat_dict = {}
    for json_path in self.path_list:
      data_dict = self._json2dict(json_path)
      self.visual_feat_dict.update(data_dict)
  
  def _updata_sample(self, rawdict):
    new_dict = {}
    for key in self.papi_feats:
      new_dict[key] = rawdict[key]
    for key in self.mfcc_egemaps:
      vowel, feat_name = key.split('_')[0], key.split('_')[1]
      if 'mfcc' == feat_name:
        filename = rawdict[key].split('.')[0]
        if filename not in self.visual_feat_dict:
          visual_feat = self.visual_feats
        else:
          visual_feat = self.visual_feat_dict[filename]
        for key, value in visual_feat.items():
          new_dict[vowel+ '-' +key] = value
    for key in self.mfcc_egemaps:
      new_dict[key] = rawdict[key]
    for key in self.labels:
      new_dict[key] = rawdict[key]
    return new_dict
  
  def update_csv(self, csv_path, new_csv_path, multi_num):
    raw_acu_feats = pd.read_csv(csv_path)
    list_of_dicts = raw_acu_feats.to_dict(orient='records')
    results = []
    if multi_num > 1:
      MP = MultiProcess()
      results = MP.multi_with_result(func=self._updata_sample, arg_list=list_of_dicts, process_num=multi_num)
    else:
      for rawdict in list_of_dicts:
        res = self._updata_sample(rawdict)
        results.append(res)
    df = pd.DataFrame(results)
    df.to_csv(new_csv_path, index=False)

