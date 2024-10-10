#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 graph_vowel_space.py
* @Time 	:	 2024/01/19
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import pandas as pd


def drop_diphthong(df_data):
  def judge_diphthong(text):
    vowels = ['a', 'o', 'e', 'i', 'u', 'v']
    vowel = ''
    for ch in text:
      if ch in vowels:
        vowel += ch
    if len(vowel) > 1: # 双元音, 多元音
      return True
    else:
      return False
  drop_index_list = []
  for index in df_data.index:
    text = df_data.iloc[index]['TEXT']
    if judge_diphthong(text):
      drop_index_list.append(index)
  df_clear = df_data.drop(drop_index_list)
  return df_clear

class VowelSpace():
  def __init__(self) -> None:
    self.vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']

  def read_csv(self, csv_path):
    basefeats = pd.read_csv(csv_path)
    basefeats = basefeats[['Person', 'TEXT', 'F1_mean', 'F2_mean', 'F3_mean']]
    mono_feat = drop_diphthong(basefeats)
    dysa_data = mono_feat[mono_feat['Person'].str.contains('S', case=False, na=False)]
    norm_data = mono_feat[mono_feat['Person'].str.contains('N', case=False, na=False)]
    return dysa_data, norm_data
  
  def delete_outline(self, vowel_data):
    from scipy.stats import zscore
    threshold = 3
    for col in ['F1_mean', 'F2_mean', 'F3-F2']:
      z_scores_col = zscore(vowel_data[col])
      outliers_mask = (abs(z_scores_col) < threshold)
      vowel_data = vowel_data[outliers_mask]
    return vowel_data

  def vowel_data_split(self, dyno_data, dy_flag):
    vowel_data_dict = {}
    for vowel in self.vowel_list:
      vowel_data = dyno_data[dyno_data['TEXT'].str.contains(vowel, case=False, na=False)]
      vowel_data['F3-F2'] = vowel_data['F3_mean'] - vowel_data['F2_mean']
      vowel_data = vowel_data.drop('F3_mean', axis=1) # 删除F3_mean
      vowel_data = self.delete_outline(vowel_data) # 删除双元音
      vowel_data_dict[vowel] = vowel_data
      vowel_data_path = '{}_{}_data.csv'.format(dy_flag, vowel)
      vowel_data.to_csv(vowel_data_path, index=False)
    return vowel_data_dict

  def plot_vowel_space(self, vowel_data_dict, dy_flag):
    for vowel, data in vowel_data_dict.items():
      men_list = list(data.mean())
      print(dy_flag, vowel, men_list[0], men_list[1], men_list[2])

if __name__ == "__main__":
  csv_path = 'tmp/pitch_ac-acousticfeat_0922.csv'
  VS = VowelSpace()
  dysa_data, norm_data = VS.read_csv(csv_path)

  n_vowel_data_dict = VS.vowel_data_split(norm_data, 'N')
  s_vowel_dict = VS.vowel_data_split(dysa_data, 'S')
  
  VS.plot_vowel_space(n_vowel_data_dict, 'N')
  VS.plot_vowel_space(s_vowel_dict, 'S')
