#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 loop_vowels.py
* @Time 	:	 2023/02/09
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import random
import itertools
random.seed(0)

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

def get_ui_loop_data(df_data):
  df_data = sel_file(df_data)
  loop_data = []
  a_data = list(df_data[df_data['TEXT'].str.contains('a')].T.to_dict().values())
  u_data = list(df_data[df_data['TEXT'].str.contains('u')].T.to_dict().values())
  i_data = list(df_data[df_data['TEXT'].str.contains('i')].T.to_dict().values())
  e_data = list(df_data[df_data['TEXT'].str.contains('e')].T.to_dict().values())
  o_data = list(df_data[df_data['TEXT'].str.contains('o')].T.to_dict().values())
  v_data = list(df_data[df_data['TEXT'].str.contains('v')].T.to_dict().values())
  for list_data in [a_data, o_data, e_data, i_data, u_data, v_data]:
    if len(list_data) <= 1:
      return loop_data
  index_a, index_e, index_o, index_v = 0, 0, 0, 0
  for sample in itertools.product(u_data, i_data):
    vowel_file_dict = {}
    vowel_file_dict['u'] = sample[0]
    vowel_file_dict['i'] = sample[1]
    vowel_file_dict['a'] = a_data[index_a]
    vowel_file_dict['e'] = e_data[index_e]
    vowel_file_dict['o'] = o_data[index_o]
    vowel_file_dict['v'] = v_data[index_v]
    loop_data.append(vowel_file_dict)
    index_a = 0 if index_a == len(a_data)-1 else index_a+1
    index_e = 0 if index_e == len(e_data)-1 else index_e+1
    index_o = 0 if index_o == len(o_data)-1 else index_o+1
    index_v = 0 if index_v == len(v_data)-1 else index_v+1
  return loop_data

def get_aoeiuv_loop_data(df_data):
  loop_data = []
  a_data = list(df_data[df_data['TEXT'].str.contains('a')].T.to_dict().values())
  u_data = list(df_data[df_data['TEXT'].str.contains('u')].T.to_dict().values())
  i_data = list(df_data[df_data['TEXT'].str.contains('i')].T.to_dict().values())
  e_data = list(df_data[df_data['TEXT'].str.contains('e')].T.to_dict().values())
  o_data = list(df_data[df_data['TEXT'].str.contains('o')].T.to_dict().values())
  v_data = list(df_data[df_data['TEXT'].str.contains('v')].T.to_dict().values())
  for list_data in [a_data, o_data, e_data, i_data, u_data, v_data]:
    if len(list_data) <= 5:
      return loop_data
  index, index_u, index_i, index_a, index_e, index_o, index_v = 0, 0, 0, 0, 0, 0, 0
  target_length = len(u_data)*len(i_data)
  if target_length < 30000: # 以防数据过多计算不便
    target_length = len(a_data)*len(u_data)*len(i_data)
  if df_data.loc[1, 'Person'].split('_')[0] == 'N': # 116分过多
    target_rate = 3000/target_length # 116分抽取3千数据
  else:
    target_rate = 30000/target_length # 抽取3万数据
  while index < target_length:
    if random.random() <= target_rate:
      vowel_file_dict = {}
      vowel_file_dict['u'] = u_data[index_u]
      vowel_file_dict['i'] = i_data[index_i]
      vowel_file_dict['a'] = a_data[index_a]
      vowel_file_dict['e'] = e_data[index_e]
      vowel_file_dict['o'] = o_data[index_o]
      vowel_file_dict['v'] = v_data[index_v]
      loop_data.append(vowel_file_dict)
      index_u = 0 if index_u == len(u_data)-1 else index_u+1
      index_i = 0 if index_i == len(i_data)-1 else index_i+1
      index_a = 0 if index_a == len(a_data)-1 else index_a+1
      index_e = 0 if index_e == len(e_data)-1 else index_e+1
      index_o = 0 if index_o == len(o_data)-1 else index_o+1
      index_v = 0 if index_v == len(v_data)-1 else index_v+1
    index += 1
  return loop_data

def get_vowels_loop_data(df_data, vowel_list, tar_num):
  loop_data, len_list = [], []
  vowel_data_dict, index_dict = {}, {}
  for vowel in vowel_list:
    data_list = list(df_data[df_data['TEXT'].str.contains(vowel)].T.to_dict().values())
    if len(data_list) < 1:
      print(df_data['Person'], vowel, 'None')
      return loop_data
    elif len(data_list) == 1:
      data_list.append(data_list[0])
    vowel_data_dict[vowel] = {'len':len(data_list), 'list':data_list}
    len_list.append(len(data_list))
    index_dict[vowel] = 0

  len_list = sorted(len_list, reverse=True)
  target_length = len_list[0] * len_list[1]
  for lenght in len_list:
    target_length = target_length*lenght
    if target_length > tar_num:
      break

  target_rate = 1
  person = df_data.loc[1, 'Person']
  if person.split('_')[0] == 'N': # 116分过多
    target_rate = (tar_num/20)/target_length
  else:
    target_rate = tar_num/target_length
  
  index = 0
  while index < target_length:
    if random.random() <= target_rate:
      vowel_file_dict = {}
      for vowel in vowel_list:
        vowel_file_dict[vowel] =  vowel_data_dict[vowel]['list'][index_dict[vowel]]
      loop_data.append(vowel_file_dict)
      for vowel in vowel_list:
        if index_dict[vowel] == vowel_data_dict[vowel]['len'] -1:
          index_dict[vowel] = 0
        else:
          index_dict[vowel] += 1
    index += 1
  return loop_data
