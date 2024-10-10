#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 acoustic_articulation_feats.py
* @Time 	:	 2022/12/01
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 extract the articulation acoustic features, GNE, CFER, tongue distance, jaw distance, movement degree(F2I/F2U ration), CSA, FCR, VAI, cariability of F1 and F2
'''

import os
import csv
import pandas as pd
from utils.multi_process import MultiProcess
from utils.normalization import Normalization
from local.loop_vowels import get_vowels_loop_data, drop_diphthong, get_ui_loop_data
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',80)

class Articulate_Feats():
  def __init__(self, sel_num, com_norm=False):
    self.sel_num = sel_num
    self.vowels = ['a','o','e','i','u', 'v']
    self.common_feats = ['Jitter', 'Shimmer', 'HNR', 'gne', 'vfer', 'F1_sd', 'F2_sd', 'F3_sd', 'Intensity_mean', 'Intensity_sd', 'Vowel_dur', 'Syllable_dur', 'gop_con','gop_vow']
    self.base_feats = ['mfcc', 'egemaps']
    self.person_feat_dict = {} # 以人来分的特征
    self.syll_com_dict = {}
    self.com_norm = com_norm
    self.Norm = Normalization()
    self.person_label = {}

  def _read_label(self, label_csv):
    # 获取每个人的frenchay分数
    with open(label_csv, encoding='gb18030') as lfp:
      reader = csv.DictReader(lfp)
      for row in reader:
        label_dict = {'Frenchay':row['Frenchay'],'fanshe':row['fanshe'], 'huxi':row['huxi'], 'chun':row['chun'], 'he':row['he'], 'ruane':row['ruane'], 'hou':row['hou'], 'she':row['she'], 'yanyu':row['yanyu']}
        self.person_label[row['Person']] = label_dict
  
  def _get_base_feat(self, vowel_file_dict):
    # 获取基础特征，比如mfcc、egemaps
    base_feat_dict = {}
    for vowel, feat_dict in vowel_file_dict.items():
      for basefeat in self.base_feats:
        vo_ba_feat = vowel + '_' + basefeat
        feat_value = os.path.basename(feat_dict[basefeat])
        base_feat_dict[vo_ba_feat] = feat_value
    return base_feat_dict

  def _articulation_compute(self, vowel_file_dict):
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
    tougue_dist = float(vowel_file_dict['i']['F2_mean']) - float(vowel_file_dict['u']['F2_mean'])
    jaw_dist = float(vowel_file_dict['a']['F1_mean']) - float(vowel_file_dict['i']['F1_mean'])
    move_degree = float(vowel_file_dict['i']['F2_mean']) / float(vowel_file_dict['u']['F2_mean'])
    VSA = abs(float(vowel_file_dict['i']['F1_mean'])*(float(vowel_file_dict['a']['F2_mean'])-float(vowel_file_dict['u']['F2_mean'])) + float(vowel_file_dict['a']['F1_mean'])*(float(vowel_file_dict['u']['F2_mean'])-float(vowel_file_dict['i']['F2_mean'])) + float(vowel_file_dict['u']['F1_mean'])*(float(vowel_file_dict['i']['F2_mean'])-float(vowel_file_dict['a']['F2_mean'])))
    VAI = (float(vowel_file_dict['i']['F2_mean']) + float(vowel_file_dict['a']['F1_mean']))/(float(vowel_file_dict['i']['F1_mean']) + float(vowel_file_dict['u']['F1_mean']) + float(vowel_file_dict['u']['F2_mean']) + float(vowel_file_dict['a']['F2_mean']))
    FCR = (float(vowel_file_dict['i']['F1_mean']) + float(vowel_file_dict['u']['F1_mean']) + float(vowel_file_dict['u']['F2_mean']) + float(vowel_file_dict['a']['F2_mean']))/(float(vowel_file_dict['i']['F2_mean']) + float(vowel_file_dict['a']['F1_mean']))
    arti_feat = {'tougue_dist':tougue_dist, 'jaw_dist':jaw_dist, 'move_degree':move_degree, 'VSA':VSA, 'VAI':VAI, 'FCR':FCR}
    if tougue_dist < 0:
      print(vowel_file_dict['i']['Filename'], vowel_file_dict['u']['Filename'])
    return arti_feat

  def _get_common_feats(self, vowel_file_dict):
    '''
     @func get_common_feats
     @desc 获取常规特征，没有特定音素限制
     @param {每条数据包含六个元音字典，每个字典包含一个基础特征字典} 
     @return {常规特征字典} 
    '''
    vowel_com_feat = {}
    for vowel, feat_dict in vowel_file_dict.items():
      for comfeat in self.common_feats:
        vo_co_feat = vowel + '-' + comfeat
        feat_value = float(feat_dict[comfeat])
        if self.com_norm:
          norm_sylla_dict = self.syll_com_dict[feat_dict['TEXT']][comfeat]
          feat_value = (feat_value - norm_sylla_dict['mean']) / norm_sylla_dict['std']
        vowel_com_feat[vo_co_feat] = feat_value
    return vowel_com_feat

  def _task_enamble(self, vowel_file_dict):
    '''
     @func _task_enamble
     @desc 
     @param {每条数据包含六个元音字典，每个字典包含一个基础特征字典}  
     @return {计算之后的特征，包括基础特征、二级特征、标签} 
    '''
    single_feat = {}
    person = vowel_file_dict['a']['Person']
    single_feat['Person'] = person
    com_feat = self._get_common_feats(vowel_file_dict)
    arti_feat = self._articulation_compute(vowel_file_dict)
    base_feat = self._get_base_feat(vowel_file_dict)
    label_dict = self.person_label[person]
    single_feat.update(arti_feat)
    single_feat.update(com_feat)
    single_feat.update(base_feat)
    single_feat.update(label_dict)
    return single_feat
  
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

  def _sample_feat(self, person_data, multi_num=1):
    '''
     @func _sample_feat
     @desc 多线程或者单线程调度，将数据分为，list的字典
           每条数据包含六个元音字典，每个字典包含一个基础特征字典
     @param {每个人的基础特征，线程数} 
     @return {每个人的声学特征} 
    '''
    complete_feats_list = []
    person_data = person_data.dropna(axis=0) # 删除所有的不标准数据
    clear_data = drop_diphthong(person_data).reset_index(drop=True) # 删除双元音
    para_list = get_vowels_loop_data(clear_data, self.vowels, self.sel_num) # loop扩充数据
    if len(para_list) == 0:
      return complete_feats_list
    if multi_num == 1:
      for vowel_file_dict in para_list:
        single_feat = self._task_enamble(vowel_file_dict) # 特征聚合在一起
        complete_feats_list.append(single_feat)
    else:
      MP = MultiProcess()
      complete_feats_list = MP.multi_with_result(func=self._task_enamble, arg_list=para_list, process_num=multi_num)
    # complete_feats_list= self._feats_delete(complete_feats_list)
    return complete_feats_list
  
  def acou_arti_interface(self, label_csv, base_feat_path, final_feat_path, multi_num):
    '''
     @func acou_arti_interface
     @desc 语音发音特征提取类的接口函数，读取基础特征计算最终的声学特征
     @param {基础特征字典，输出特征csv地址，线程数}  
     @return {空}
    '''
    feat_dict_list = []
    basefeats = pd.read_csv(base_feat_path) # 读取base特征
    self._read_label(label_csv) # 读取标签
    if self.com_norm:
      self.syll_com_dict = self.Norm.lxk_common_sylla(basefeats) # 归一化方法
    persons = basefeats['Person'].unique()
    print(len(persons))
    problem_person = ['S_00040_M', 'S_00037_M'] # 音素v缺失
    for p in persons:
      if p in problem_person:
        continue
      person_data = basefeats[basefeats['Person'] == p].reset_index(drop=True)
      person_feats_list = self._sample_feat(person_data, multi_num)
      if len(person_feats_list) == 0:
        continue
      feat_dict_list.extend(person_feats_list)
    
    df = pd.DataFrame(feat_dict_list)
    df.to_csv(final_feat_path, index=False)


if __name__ == "__main__":
  label_csv = 'data/Label.csv'
  base_feat_path = 'tmp/pitch_ac-acousticfeat_0803.csv'
  final_feat_path = 'data/result_intermediate/acoustic_loop_feats_0922.csv'
  sel_num = 30000
  AF = Articulate_Feats(sel_num, com_norm=False)
  AF.acou_arti_interface(label_csv, base_feat_path, final_feat_path, 80)
