#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 feat_select.py
* @Time 	:	 2023/08/15
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025'','' lxk&AISML
* @Desc   	:	 None
'''
import os
import pandas as pd
import datetime

common_feats = ['Jitter','Shimmer','HNR','gne','vfer','F1_sd','F2_sd','F3_sd','Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur','gop_con','gop_vow']
arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']

gop_feats = ['gop_con','gop_vow']
LiuYY_arti_feats = ['move_degree','VSA','VAI','FCR']

l_fets = ['alpha_stability','beta_stability','dist_stability','alpha_speed','beta_speed','inner_speed','inner_dist_min','inner_dist_max','w_min','w_max']

labels = ['Frenchay','fanshe','huxi','chun','he','ruane','hou','she','yanyu']
# labels = ['Frenchay','fanshe','huxi','chun','he','ruane','hou','she','yanyu','ljri', 'rlvt']

segment_feats = ['Segname']
srft_feats = ['Path','Start','End']


def get_feat_name(feat_type, vowels):
  sel_feats_name = ['Person']
  if feat_type == 'loop_featnorm' or feat_type == 'papi' or feat_type == 'gop_loop_featnorm' or feat_type == 'phonation' or feat_type == 'articulation' or feat_type == 'prosody':
    vowel_common = []
    for vowel in vowels:
      for feat in common_feats:
        vowel_common.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + arti_feats + vowel_common + labels
  elif feat_type == 'lip_cmlrv':
    vowel_lip, vowel_mfcc = [], []
    for vowel in vowels:
      for feat in l_fets:
        vowel_lip.append(vowel + '-' + feat)
      vowel_mfcc.append(vowel + '_' + 'mfcc')
    sel_feats_name = sel_feats_name + vowel_lip + vowel_mfcc + labels
  elif feat_type == 'papi_cmrlv':
    vowel_common, vowel_mfcc = [], []
    for vowel in vowels:
      for feat in common_feats:
        vowel_common.append(vowel + '-' + feat)
      vowel_mfcc.append(vowel + '_' + 'mfcc')
    sel_feats_name = sel_feats_name + arti_feats + vowel_common + vowel_mfcc + labels
  elif feat_type == 'papi_lip':
    vowel_lip, vowel_common = [], []
    for vowel in vowels:
      for feat in l_fets:
        vowel_lip.append(vowel + '-' + feat)
      for feat in common_feats:
        vowel_common.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + arti_feats + vowel_common + vowel_lip + labels
  elif feat_type == 'papi_lipcmrlv':
    vowel_lip, vowel_common, vowel_mfcc = [], [], []
    for vowel in vowels:
      for feat in l_fets:
        vowel_lip.append(vowel + '-' + feat)
      for feat in common_feats:
        vowel_common.append(vowel + '-' + feat)
      vowel_mfcc.append(vowel + '_' + 'mfcc')
    sel_feats_name = sel_feats_name + arti_feats + vowel_common + vowel_lip + vowel_mfcc + labels
  elif feat_type == 'loop_v_featnorm' or feat_type == 'lip':
    vowel_lip = []
    for vowel in vowels:
      for feat in l_fets:
        vowel_lip.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + vowel_lip + labels
  elif feat_type == 'loop_av_featnorm':
    vowel_papil = []
    for vowel in vowels:
      for feat in common_feats:
        vowel_papil.append(vowel + '-' + feat)
      for feat in l_fets:
        vowel_papil.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + arti_feats + vowel_papil + labels
  elif feat_type == 'mfcc':
    vowel_mfcc = []
    for vowel in vowels:
      vowel_mfcc.append(vowel + '_' + 'mfcc')
    sel_feats_name = sel_feats_name + vowel_mfcc + labels
  elif feat_type == 'egemaps':
    vowel_egemaps = []
    for vowel in vowels:
      vowel_egemaps.append(vowel + '_' + 'egemaps')
    sel_feats_name = sel_feats_name + vowel_egemaps + labels
  elif feat_type == 'gop':
    vowel_gop = []
    for vowel in vowels:
      for feat in gop_feats:
        vowel_gop.append(vowel + '-' + feat)
    sel_feats_name = sel_feats_name + vowel_gop + labels
  elif feat_type == 'Liuartifeat':
    sel_feats_name = sel_feats_name + LiuYY_arti_feats + labels
  elif feat_type == 'stft':
    sel_feats_name = sel_feats_name + srft_feats + labels
  elif feat_type == 'papi_cmlrv' or feat_type == 'papi_cropavi':
    vowel_papil = []
    vowel_mfcc = []
    for vowel in vowels:
      for feat in common_feats:
        vowel_papil.append(vowel + '-' + feat)
    for vowel in vowels:
      vowel_mfcc.append(vowel + '_' + 'mfcc')
    sel_feats_name = sel_feats_name + arti_feats + vowel_papil + vowel_mfcc + labels
  elif feat_type == 'cqcc':
    vowel_cqcc = []
    for vowel in vowels:
      vowel_cqcc.append(vowel + '_' + 'cqcc')
    sel_feats_name = sel_feats_name + vowel_cqcc + labels
  elif 'segment' in feat_type:
    sel_feats_name = sel_feats_name + segment_feats + labels
  return sel_feats_name

def feat_sel(feat_type, vowels, featsdf):
  feats_name = get_feat_name(feat_type, vowels)
  sel_feat = featsdf[feats_name]
  return sel_feat

def score2class(label, featsdf, class_num=4):
  featsdf.loc[(featsdf[label] >= 29) & (featsdf[label] <= 57), label] = 3 # 9000
  featsdf.loc[(featsdf[label] >= 58) & (featsdf[label] <= 86), label] = 2 # 14000
  featsdf.loc[(featsdf[label] >= 87) & (featsdf[label] <= 115), label] = 1 # 38000
  featsdf.loc[(featsdf[label] == 116), label] = 0 # 4500
  return featsdf

def get_file_path(segment_dir, file_name, suffix):
  name_ll = file_name.split('_')
  if len(name_ll) == 7:
    pass
  elif len(name_ll) == 8:
    name_ll = name_ll[1:]
  else:
    print(file_name, 'error file')
    exit(-1) 
  N_S_name = "Control" if name_ll[0] == "N" else "Patient"
  person_name = name_ll[0] + '_' + name_ll[2] + '_' + name_ll[1]
  file_path = os.path.join(segment_dir, N_S_name, person_name, file_name + suffix)
  return file_path


def NullDelete(feat_type, feat_data, file_dir):
  if 'wav2vec' in feat_type:
    suffix = '.w2v_max.pt'
  elif feat_type == 'vhubertsegment':
    suffix = '.npy'
  elif feat_type == 'hubertsegment':
    suffix = '.npy'
  elif 'cqcc' in feat_type:
    suffix = '.cqcc.pt'
  elif 'ivector' in feat_type:
    suffix = '.ivector.pt'
  elif 'vhubert' in feat_type:
    suffix = '.vhubert.npy'
  elif 'cropavi' in feat_type:
    suffix = '.crop.pt'
  else:
    return feat_data

  # judge whether mfcc exist in columns
  if 'segment' in feat_type:
    filtered_columns = feat_data.filter(like='Segname', axis=1).columns
  else:
    filtered_columns = feat_data.filter(like='mfcc', axis=1).columns
  # 如果关于mfcc的列filtered_columns为空，说明是段级别的特征
  if len(filtered_columns) == 0:
    print('NullDelete: No columns in the dataframe')
    exit(-1)
  row_dict_list = feat_data.T.to_dict().values()
  new_row_list = []
  for row_dict in row_dict_list:
    file_exist_flag = True
    for colname in filtered_columns:
      if 'segment' in feat_type: # 段级别的特征
        mfcc_name = row_dict[colname].split('.')[0]
        file_path = os.path.join(file_dir, mfcc_name + suffix)
      else: # 音节级别的特征
        mfcc_name = row_dict[colname].split('.')[0]
        file_path = get_file_path(file_dir, mfcc_name, suffix)
      if not os.path.exists(file_path):
        file_exist_flag = False
        continue
    if file_exist_flag:
      new_row_list.append(row_dict)
  new_dataframe = pd.DataFrame(new_row_list).reset_index(drop=True)
  date_str = datetime.datetime.now().strftime('%Y-%m-%d')
  # new_dataframe.to_csv('tmp/'+date_str+feat_type+'.csv', index=False)
  return new_dataframe
