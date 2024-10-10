#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 vowel_rhythm.py
* @Time 	:	 2023/12/19
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import numpy as np
import pandas as pd

def Tone():
  pass

def SeverityLevel():
  pass

def read_task2word(task2word):
  task_class = {}
  with open(task2word, 'r') as fp:
    for line in fp:
      line = line.rstrip('\n')
      ll = line.split(' ')
      mean_len = round(sum(len(s) for s in ll[1:])/len(ll[1:]))
      # if mean_len > 3 :
      #   task_class[ll[0]] = 3 # sentence
      # elif mean_len > 3 :
      #   task_class[ll[0]] = 2 # word
      # elif mean_len > 1 :
      #   task_class[ll[0]] = 2 # word
      # else:
      #   task_class[ll[0]] = 1 # syllabel
      task_class[ll[0]] = mean_len
  return task_class

def name_reshape(filename):
  filename = filename.split('.')[0]
  ll = filename.split('G')[-1].split('_')
  new_name = 'G'+ ll[0] + '_' + ll[1] + '_' + ll[2]
  return new_name

def read_result_value(resultvalue, task_class):
  feat_data = pd.read_csv(resultvalue)
  feat_data = feat_data[['a_mfcc', 'o_mfcc', 'e_mfcc', 'i_mfcc','u_mfcc', 'v_mfcc', 'Frenchay', 'Predict']]
  score_value_dict = {}
  row_dict_list = feat_data.T.to_dict().values()
  for feat_dict in row_dict_list:
    score = 0
    for filename in feat_dict.keys():
      if 'mfcc' not in filename:
        continue
      name = name_reshape(feat_dict[filename])
      score += task_class[name]
    d_value = abs(feat_dict['Predict'] - feat_dict['Frenchay'])
    if score > 18:
      score = 3
    elif score > 6:
      score = 2
    else:
      score = 1
    if score not in score_value_dict:
      score_value_dict[score] = [d_value]
    else:
      score_value_dict[score].append(d_value)
  for key, value in score_value_dict.items():
    value_mean = np.mean(value)
    value_std = np.std(value)
    result = {'score':key, 'mean':value_mean, 'std':value_std, 'count':len(value)}
    print(result)


if __name__ == "__main__":
  resultvalue = 'tmp/rhythm_test/Frenchay_value.csv'
  task2word = 'tmp/rhythm_test/Task2Word.txt'
  task_class = read_task2word(task2word)
  read_result_value(resultvalue, task_class)

