#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 acoustic_lsj_feats.py
* @Time 	:	 2023/02/02
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import pandas as pd
import numpy as np
import itertools
import json
import csv
from utils.multi_process import MultiProcess
import warnings
warnings.filterwarnings('ignore')

class ArticulateLSJFeats():
  def __init__(self):
    self.vowels = ['a','o','e','i','u','v']
    self.person_feat_dict = {} # 以人来分的特征
    self.temple_feats = {'TEXT':None, 'Person':None, 'tougue_dist':None, 'jaw_dist':None, 'move_degree':None, 'VSA':None, 'VAI':None, 'FCR':None, 'Jitter':None, 'Shimmer':None, 'HNR':None, 'gne':None, 'vfer':None, 'F1_sd':None, 'F2_sd':None, 'F3_sd':None, 'Intensity_mean':None, 'Intensity_sd':None, 'Vowel_dur':None, 'Syllable_dur':None, 'syllabel_acc':None, 'total_acc':None,'Frenchay':None,'fanshe':None, 'huxi':None, 'chun':None, 'he':None, 'ruane':None, 'hou':None, 'she':None, 'yanyu':None}
    self.syll_all_dict = {}
    self.person_label = {}

  def _add_label(self, label_csv, data_list):
    label_list = ['Frenchay','fanshe', 'huxi', 'chun', 'he', 'ruane', 'hou', 'she', 'yanyu']
    # get person label dict
    with open(label_csv) as lfp:
      reader = csv.DictReader(lfp)
      for row in reader:
        label_dict = {}
        for label in label_list:
          label_dict[label] = row[label]
        self.person_label[row['Person']] = label_dict
    
    # fill data_list
    for row in data_list:
      for label in label_list:
        row[label] = self.person_label[row['Person']][label]
    return data_list

  def _dict_copy(self, sour_dict, dest_dict):
    for key, value in sour_dict.items():
      if key in dest_dict:
        if str(value).replace('.', '1').isdigit():
          dest_dict[key] = float(value)
        else:
          dest_dict[key] = value
    return dest_dict
  
  def _get_feat_array(self, vowel_list, feat_key):
    feat_list = []
    for sample_dict in vowel_list:
      if sample_dict[feat_key] != None:
        feat_list.append(float(sample_dict[feat_key]))
    return np.array(feat_list)
  
  def _lsj_pad(self, person_data, person_feat_list):
    tougue_dist_mean = np.nanmean(self._get_feat_array(person_feat_list, 'tougue_dist'))
    jaw_dist_mean = np.nanmean(self._get_feat_array(person_feat_list, 'jaw_dist'))
    move_degree_mean = np.nanmean(self._get_feat_array(person_feat_list, 'move_degree'))
    VSA_mean = np.nanmean(self._get_feat_array(person_feat_list, 'VSA'))
    VAI_mean = np.nanmean(self._get_feat_array(person_feat_list, 'VAI'))
    FCR_mean = np.nanmean(self._get_feat_array(person_feat_list, 'FCR'))
    arti_feat_dict = {'tougue_dist':tougue_dist_mean, 'jaw_dist':jaw_dist_mean, 'move_degree':move_degree_mean, 'VSA':VSA_mean, 'VAI':VAI_mean, 'FCR':FCR_mean}

    for sample_feat in person_feat_list:
      for key, value in arti_feat_dict.items():
        if sample_feat[key] == None:
          sample_feat[key] = value
    
    for vowel in ['o', 'e', 'v']:
      vowel_data = list(person_data[person_data['TEXT'].str.contains(vowel)].T.to_dict().values())
      for sample in vowel_data:
        sample_feat = self.temple_feats.copy()
        sample_feat['tougue_dist'] = tougue_dist_mean
        sample_feat['jaw_dist'] = jaw_dist_mean
        sample_feat['move_degree'] = move_degree_mean
        sample_feat['VSA'] = VSA_mean
        sample_feat['VAI'] = VAI_mean
        sample_feat['FCR'] = FCR_mean
        sample_feat = self._dict_copy(sample, sample_feat)
        person_feat_list.append(sample_feat)
    return person_feat_list
  
  def _articulation_compute(self, person_data):
    '''
     @func 
     @desc 
     @param {每条数据包含六个元音字典，每个字典包含一个基础特征字典}
     @return {返回特征字典} 
      tougue_dist : F2(i) - F2(u)
      jaw_dist  :   F1(a) - F1(i)
      move_degree :   F2(i) / F2(u)
      VSA :   |F1(i)*[F2(a)-F2(u)] + F1(a)*[F2(u)-F2(i)] + F1(u)*[F2(i)-F2(a)]|/2
      VAI :   [F2(i) + F1(a)] / [F1(i) + F1(u) + F2(u) + F2(a)]
      FCR :   [F1(i) + F1(u) + F2(u) + F2(a)] / [F2(i) + F1(a)]
    '''
    a_data = list(person_data[person_data['TEXT'].str.contains('a')].T.to_dict().values())
    u_data = list(person_data[person_data['TEXT'].str.contains('u')].T.to_dict().values())
    i_data = list(person_data[person_data['TEXT'].str.contains('i')].T.to_dict().values())

    person_feat_list = []
    # 元音a提取特征VSA， VAI， FCR
    for sample in a_data:
      sample_feat = self.temple_feats.copy()
      VSA = []; VAI = []; FCR = []
      list_i = list(range(len(i_data)))
      list_u = list(range(len(u_data)))
      for order in list(itertools.product(list_i, list_u)):
        F1_a = float(sample['F1_mean'])
        F2_a = float(sample['F2_mean'])
        F1_i = float(i_data[order[0]]['F1_mean'])
        F2_i = float(i_data[order[0]]['F2_mean'])
        F1_u = float(u_data[order[1]]['F1_mean'])
        F2_u = float(u_data[order[1]]['F2_mean'])
        VSA.append(abs(F1_i * (F2_a - F2_u) + F1_a * (F2_u - F2_i) + F1_u * (F2_i - F2_a)) / 2.0)
        VAI.append((F2_i + F1_a) / (F1_i + F1_u + F2_u + F2_a))
        FCR.append((F1_i + F1_u + F2_u + F2_a) / (F2_i + F1_a))
      sample_feat['VSA'] = np.nanmean(VSA)
      sample_feat['VAI'] = np.nanmean(VAI)
      sample_feat['FCR'] = np.nanmean(FCR)
      sample_feat = self._dict_copy(sample, sample_feat)
      person_feat_list.append(sample_feat)
    
    # 元音i提取特征tougue_dist, jaw_dist, move_degree, VSA， VAI， FCR
    for sample in i_data:
      sample_feat = self.temple_feats.copy()
      sample_feat['tougue_dist'] = np.nanmean(float(sample['F2_mean']) - self._get_feat_array(u_data, 'F2_mean'))
      sample_feat['jaw_dist'] = np.nanmean(self._get_feat_array(a_data, 'F1_mean') - float(sample['F1_mean']))
      sample_feat['move_degree'] = np.nanmean(float(sample['F2_mean'])/self._get_feat_array(u_data, 'F2_mean'))
      VSA = []; VAI = []; FCR = []
      list_a = list(range(len(a_data)))
      list_u = list(range(len(u_data)))
      for order in list(itertools.product(list_a, list_u)):
        F1_a = float(a_data[order[0]]['F1_mean'])
        F2_a = float(a_data[order[0]]['F2_mean'])
        F1_i = float(sample['F1_mean'])
        F2_i = float(sample['F2_mean'])
        F1_u = float(u_data[order[1]]['F1_mean'])
        F2_u = float(u_data[order[1]]['F2_mean'])
        VSA.append(abs(F1_i * (F2_a - F2_u) + F1_a * (F2_u - F2_i) + F1_u * (F2_i - F2_a)) / 2.0)
        VAI.append((F2_i + F1_a) / (F1_i + F1_u + F2_u + F2_a))
        FCR.append((F1_i + F1_u + F2_u + F2_a) / (F2_i + F1_a))
      sample_feat['VSA'] = np.nanmean(VSA)
      sample_feat['VAI'] = np.nanmean(VAI)
      sample_feat['FCR'] = np.nanmean(FCR)
      sample_feat = self._dict_copy(sample, sample_feat)
      person_feat_list.append(sample_feat)
    
    # 元音u提取特征tougue_dist, move_degree, VSA， VAI， FCR
    for sample in u_data:
      sample_feat = self.temple_feats.copy()
      sample_feat['tougue_dist'] = np.nanmean(self._get_feat_array(i_data, 'F2_mean') - float(sample['F2_mean']))
      sample_feat['move_degree'] = np.nanmean(self._get_feat_array(i_data, 'F2_mean')/float(sample['F2_mean']))
      VSA = []; VAI = []; FCR = []
      list_a = list(range(len(a_data)))
      list_i = list(range(len(i_data)))
      for order in list(itertools.product(list_a, list_i)):
        F1_a = float(a_data[order[0]]['F1_mean'])
        F2_a = float(a_data[order[0]]['F2_mean'])
        F1_u = float(i_data[order[1]]['F1_mean'])
        F2_u = float(i_data[order[1]]['F2_mean'])
        F1_i = float(sample['F1_mean'])
        F2_i = float(sample['F2_mean'])
        VSA.append(abs(F1_i * (F2_a - F2_u) + F1_a * (F2_u - F2_i) + F1_u * (F2_i - F2_a)) / 2.0)
        VAI.append((F2_i + F1_a) / (F1_i + F1_u + F2_u + F2_a))
        FCR.append((F1_i + F1_u + F2_u + F2_a) / (F2_i + F1_a))
      sample_feat['VSA'] = np.nanmean(VSA)
      sample_feat['VAI'] = np.nanmean(VAI)
      sample_feat['FCR'] = np.nanmean(FCR)
      sample_feat = self._dict_copy(sample, sample_feat)
      person_feat_list.append(sample_feat)
    
    # 元音a填充tougue_dist, jaw_dist, move_degree特征
    # 元音u填充特征jaw_dist
    # 元音oev填充tougue_dist, jaw_dist, move_degree, VSA， VAI， FCR特征
    person_feat_list = self._lsj_pad(person_data, person_feat_list)
    return person_feat_list
  
  def _feats_delete(self, complete_feats_list):
    '''
     @func _feats_delete
     @desc 对错误特征按人进行填充
     @param {特征list}
     @return {修改过后的特征list} 
    '''
    new_feats_list = []
    tougue_dist_error = 0
    jaw_dist_error = 0
    move_degree_error = 0
    tougue_dist_sum = 0
    jaw_dist_sum = 0
    move_degree_sum = 0
    error_flag = False
    feat_len = len(complete_feats_list)

    for single_feat in complete_feats_list:
      if single_feat['tougue_dist'] > 0:
        tougue_dist_sum += single_feat['tougue_dist']
      if single_feat['jaw_dist'] > 0:
        jaw_dist_sum += single_feat['jaw_dist']
      if single_feat['move_degree'] > 1:
        move_degree_sum += single_feat['move_degree']

    for single_feat in complete_feats_list:
      if single_feat['tougue_dist'] < 0:
        tougue_dist_error += 1
        single_feat['tougue_dist'] = tougue_dist_sum/feat_len
      if single_feat['jaw_dist'] < 0:
        jaw_dist_error += 1
        single_feat['jaw_dist'] = jaw_dist_sum/feat_len
      if single_feat['move_degree']<1:
        move_degree_error += 1
        single_feat['move_degree'] = move_degree_sum/feat_len
      new_feats_list.append(single_feat)
    print('tougue_dist_error', tougue_dist_error, 'jaw_dist_error', jaw_dist_error, 'move_degree_error', move_degree_error)
    print('before delete length', len(complete_feats_list), ' after delete length', len(new_feats_list))
    return new_feats_list
  
  def acou_arti_interface(self, label_csv, base_feat_path, csv_path, multi_num=1):
    '''
     @func acou_arti_interface
     @desc 语音发音特征提取类的接口函数，读取基础特征计算最终的声学特征
     @param {基础特征字典，输出特征csv地址，线程数}  
     @return {空} 
    '''
    persons_list = []
    basefeats = pd.read_csv(base_feat_path)
    persons = list(basefeats['Person'].unique())
    for p in persons:
      persons_list.append(basefeats[basefeats['Person'] == p].reset_index(drop=True))
    
    complete_feats_list = []
    if multi_num == 1:
      for person_data in persons_list:
        person_feats = self._articulation_compute(person_data)
        complete_feats_list.extend(person_feats)
    else:
      MP = MultiProcess()
      persons_feats = MP.multi_with_result(func=self._articulation_compute, arg_list=persons_list, process_num=multi_num)
      complete_feats_list = [single_feat for person_feats in persons_feats for single_feat in person_feats]

    # complete_feats_list= self._feats_delete(complete_feats_list)
    complete_feats_list = self._add_label(label_csv, complete_feats_list)
    df = pd.DataFrame(complete_feats_list)
    df.to_csv(csv_path)

if __name__ == "__main__":
  label_csv = 'data/Label.csv'
  base_feat_path = 'tmp/pitch_ac-acousticfeat_0305.csv'
  csv_path = 'data/result_intermediate/lsj_acoustic_feats_0305.csv'
  AF = ArticulateLSJFeats()
  AF.acou_arti_interface(label_csv, base_feat_path, csv_path, 60)
