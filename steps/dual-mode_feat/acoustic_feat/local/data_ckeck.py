#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 data_ckeck.py
* @Time 	:	 2022/11/22
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 数据检查，特征统计分析
'''

import os
import sys
import csv
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler


class BaseFeatCheck():
  '''
    统计pitch, F1 mean, F2 mean等基础特征
    观察每种参数是否有离群点，是否合理
  '''
  def __init__(self):
    self.pitch_dict = {}
    self.feat_dict= {}
    self.short_dict = {}
    self.MM = MinMaxScaler()

  def _read_data(self, data_dir, condition):
    for root, dirs, files in os.walk(data_dir):
      for file in files:
        if '.txt' in file and condition in file:
          data = np.loadtxt(os.path.join(root, file), delimiter= ' ')
          if len(data) <5:
            continue
          self.pitch_dict[file] = data[:,0]
        if '.csv' in file and condition in file:
          self.df_data = pd.read_csv(os.path.join(root, file))
          self.feat_dict[file.split('.')[0]] = self.df_data.dropna(inplace=False) # 删除空值
          # self.feat_dict[file.split('.')[0]] = self.df_data[self.df_data['Vowel_dur']>0.05] # 筛选长度大于0.04秒

  def _pitch_data_select(self, condition='txt'):
    sele_data = np.array([])
    for key, value in self.pitch_dict.items():
      if condition in key:
        if sele_data.size == 0:
          sele_data = value
        else:
          sele_data = np.concatenate((sele_data, value))
    return sele_data

  def _data_analysis(self, data, name):
    data_label = self.MM.fit_transform(np.expand_dims(data, axis=1))
    norm_dict = {}
    for i in range(len(data_label)):
      da = data_label[i][0]
      da_str = str(round(da, 2))
      if da_str not in norm_dict:
        norm_dict[da_str] = 1
      else:
        norm_dict[da_str] += 1

    result_dict = {}
    for key, values in norm_dict.items():
      predict_value = self.MM.inverse_transform(np.array([[float(key)]]))[0][0]
      result_dict[predict_value] = values
    
    sorted_keys = sorted(result_dict.keys())
    value_sum = len(data_label)

    with open('tmp/{}'.format(name), 'w') as fp:
      for key in sorted_keys:
        fp.write(str(key)+','+str(result_dict[key])+','+str(result_dict[key]/value_sum)+'\n')
        # print(key, result_dict[key], result_dict[key]/value_sum)
      fp.write(str(value_sum)+'\n')

  def _get_all_select_condition(self):
    select_conditions = []
    person_list = []
    pitch_type = ['pitch', 'pitch_cc', 'pitch_ac']
    with open('tmp/person', 'r') as fp:
      for line in fp:
        person = line.rstrip('\n')
        person_list.append(person)

    for pt in pitch_type:
      for person in person_list:
        condition = pt + '-' + person
        select_conditions.append(condition)
    return select_conditions

  def check_pitch(self, data_dir, read_condition='.txt'):
    self._read_data(data_dir, read_condition)
    select_conditions = self._get_all_select_condition()
    for sc in select_conditions:
      data = self._pitch_data_select(sc)
      self._data_analysis(data, 'pitch_count/'+sc)
  
  def check_csv(self, data_dir, condition):
    self._read_data(data_dir, condition)
    for name, pd_data in self.feat_dict.items():
      for col in pd_data.columns:
        if col == 'Filename' or col == 'Person' or col == 'TEXT':
          continue
        feat_data = pd_data[col].to_numpy()
        self._data_analysis(feat_data, 'feat_count/'+name+'_'+col)

class FeatSampleCheck():
  '''
    按照人来统计，元音、辅音、音节的数量
  '''
  def __init__(self, csv_path):
    self.df_data = pd.read_csv(csv_path)
    self.person_num = {}
    self.syllable_num = {} # 68个音节
    self.vowel_num = {'a':0, 'o':0, 'e':0, 'i':0, 'u':0, 'v':0} # 6个元音
    self.consonant_num = {'b':0, 'p':0, 'm':0, 'f':0, 'd':0, 't':0, 'n':0, 'l':0, 'g':0, 'k':0, 'h':0, 'j':0, 'q':0, 'x':0, 'zh':0, 'ch':0, 'sh':0, 'r':0, 'z':0, 'c':0, 's':0, 'ng':0} # 22个辅音
    self.syllable_tone_num = {'a1':0} # 119个带音调音节

  def _get_vowel(self, text):
    vowels = ['a', 'o', 'e', 'i', 'u', 'v']
    vowel, consonant, syllable = '', '', ''
    for i in range(len(text)):
      ch = text[i]
      if '0' < ch < '9':
        continue
      else:
        if ch in vowels:
          vowel += ch
        else:
          if len(vowel)==0 and i < 2:
            consonant+=ch
        syllable += ch
    return vowel, consonant, syllable

  def get_person_syllable(self, tmp_df):
    vowel_num = dict.fromkeys(self.vowel_num, 0)
    consonant_num =dict.fromkeys(self.consonant_num, 0)
    syllable_num =dict.fromkeys(self.syllable_num, 0)
    syllable_tone_num =dict.fromkeys(self.syllable_tone_num, 0)
    person_num = self.person_num

    for row in tmp_df.index:
      person=tmp_df.loc[row]['Person']
      text=tmp_df.loc[row]['TEXT']

      if text == 'a1':
        syllable_tone_num[text] += 1
        continue
      vowel, consonant, syllable = self._get_vowel(text)

      vowel_num[vowel] = vowel_num[vowel]+1 if vowel in vowel_num else 1
      consonant_num[consonant] = consonant_num[consonant]+1 if consonant in consonant_num else 1
      syllable_num[syllable] = syllable_num[syllable]+1 if syllable in syllable_num else 1
      syllable_tone_num[text] = syllable_tone_num[text]+1 if text in syllable_tone_num else 1
      person_num[person] = person_num[person]+1 if person in person_num else 1
    return vowel_num, consonant_num, syllable_num, syllable_tone_num, person_num
  
  def person_info_statistic(self, json_path):
    all_person_dict = {}
    for person in self.person_num.keys():
      df_person=self.df_data[self.df_data['Person']==person]
      vowel_num, consonant_num, syllable_num, syllable_tone_num, person_num = self.get_person_syllable(df_person)
      vowel_num.update({'person':person})
      consonant_num.update({'person':person})
      syllable_tone_num.update({'person':person})
      all_person_dict[person] = {'vowel_num':vowel_num, 'consonant_num':consonant_num, 'syllable_num':syllable_num, 'syllable_tone_num':syllable_tone_num}
    fp = open(json_path, 'w')
    fp.write(json.dumps(all_person_dict))
    fp.close()
  
  def load_json(self, json_path):
    fp = open(json_path, 'r')
    all_person_dict = json.load(fp)
    vowel_num_list = []
    consonant_num_list = []
    syllable_num_list = []
    syllable_tone_num_list = []
    for person, num_dict in all_person_dict.items():
      vowel_num_list.append(num_dict['vowel_num'])
      consonant_num_list.append(num_dict['consonant_num'])
      syllable_num_list.append(num_dict['syllable_num'])
      syllable_tone_num_list.append(num_dict['syllable_tone_num'])
      # print(person, sorted(vowel_num.items(),key=lambda item:item[1],reverse=True))
      # print(person, sorted(consonant_num.items(),key=lambda item:item[1],reverse=True))
      # print(person, sorted(syllable_num.items(),key=lambda item:item[1],reverse=True))
      # print(person, sorted(syllable_tone_num.items(),key=lambda item:item[1],reverse=True))
    vowel_num_pd = pd.DataFrame(vowel_num_list)
    consonant_num_pd = pd.DataFrame(consonant_num_list)
    syllable_num_pd = pd.DataFrame(syllable_num_list)
    syllable_tone_num_pd = pd.DataFrame(syllable_tone_num_list)
    vowel_num_pd.to_csv('tmp/vowel_count.csv', encoding='utf-8', index=False)
    consonant_num_pd.to_csv('tmp/consonant_count.csv', encoding='utf-8', index=False)
    syllable_num_pd.to_csv('tmp/syllable_count.csv', encoding='utf-8', index=False)
    syllable_tone_num_pd.to_csv('tmp/syllable_tone_count.csv', encoding='utf-8', index=False)

class LoopedFeatCheck():
  def __init__(self, label_csv) -> None:
    self.person_label = {}
    self._read_label(label_csv)

  def _read_label(self, label_csv):
    # 获取每个人的frenchay分数
    with open(label_csv, encoding='gb18030') as lfp:
      reader = csv.DictReader(lfp)
      for row in reader:
        label_dict = {'Frenchay':row['Frenchay'],'fanshe':row['fanshe'], 'huxi':row['huxi'], 'chun':row['chun'], 'he':row['he'], 'ruane':row['ruane'], 'hou':row['hou'], 'she':row['she'], 'yanyu':row['yanyu']}
        self.person_label[row['Person']] = label_dict

  def loop_feat_count(self, loop_csv):
    File = open(loop_csv, 'r')
    reader = csv.DictReader(File)
    person_dict = {}
    for dictionary in reader:
      person = dictionary['Person']
      if person in person_dict:
        person_dict[person] += 1
      else:
        person_dict[person] = 1
    for key, value in person_dict.items():
      # print(key, value)
      if key not in self.person_label:
        continue
      print(key, value, self.person_label[key]['Frenchay'])


if __name__ == "__main__":
  # data_dir = 'tmp/'
  # condition = 'N_F_10001'
  # AFC = BaseFeatCheck()
  # # AFC.check_pitch(data_dir)
  # AFC.check_csv(data_dir, 'pitch_ac-acousticfeat')

  # FSC = FeatSampleCheck('tmp/pitch_ac-acousticfeat_0625.csv')
  # FSC.vowel_num, FSC.consonant_num, FSC.syllable_num, FSC.syllable_tone_num, FSC.person_num = FSC.get_person_syllable(FSC.df_data)
  # FSC.person_info_statistic('tmp/all_person_info.json')
  # FSC.load_json('tmp/all_person_info.json')

  label_csv = 'data/Label.csv'
  data_csv = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/20230617.csv'
  base_csv = 'tmp/pitch_ac-acousticfeat_0625.csv'
  loop_csv = 'data/result_intermediate/acoustic_loop_feats_sel_3w.csv'
  LFC = LoopedFeatCheck(label_csv)
  LFC.loop_feat_count(loop_csv)
