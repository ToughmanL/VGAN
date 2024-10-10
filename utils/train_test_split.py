#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 train_test_split.py
* @Time 	:	 2023/03/02
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 在医学领域需要按照人来分测试训练集
'''
import pandas as pd

def normal_test_person():
  "增加数据之后训练集测试集分割"
  fold0 = ['N_10001_F', 'N_10011_M', 'S_00017_M', 'S_00071_M', 'S_00033_F', 'S_00057_M', 'S_00054_F', 'S_00016_M', 'N_10009_M']
  fold1 = ['N_10025_F', 'N_10012_M', 'S_00019_F', 'S_00056_F', 'S_00073_M', 'S_00058_M', 'S_00045_M', 'S_00010_F', 'N_10014_M']
  fold2 = ['N_10023_M', 'N_10013_M', 'S_00059_M', 'S_00063_M', 'S_00069_M', 'S_00076_F', 'S_00008_M', 'S_00034_M', 'N_10003_M']
  fold3 = ['N_10004_M', 'N_10022_F', 'S_00035_M', 'S_00032_M', 'S_00052_F', 'S_00020_M', 'S_00003_M', 'S_00012_M', 'N_10002_F']
  fold4 = ['N_10005_F', 'N_10015_M', 'S_00051_M', 'S_00070_M', 'S_00072_M', 'S_00068_M', 'S_00027_F', 'S_00013_M', 'N_10010_M']
  fold5 = ['N_10006_F', 'N_10016_M', 'S_00077_F', 'S_00062_M', 'S_00078_M', 'S_00026_M', 'S_00049_F', 'S_00021_F']
  fold6 = ['N_10007_F', 'N_10017_M', 'S_00075_M', 'S_00067_F', 'S_00043_M', 'S_00047_M', 'S_00061_M', 'S_00009_M']
  fold7 = ['N_10008_F', 'N_10018_F', 'S_00060_M', 'S_00046_F', 'S_00065_M', 'S_00044_M', 'S_00031_F', 'S_00014_M']
  fold8 = ['N_10021_F', 'N_10019_M', 'S_00066_F', 'S_00024_M', 'S_00030_M', 'S_00005_M', 'S_00055_M', 'S_00015_F']
  fold9 = ['N_10024_F', 'N_10020_F', 'S_00022_M', 'S_00064_F', 'S_00004_M', 'S_00053_M', 'S_00050_F', 'S_00023_M']
  # N_10009_M N_10014_M N_10003_M N_10002_F N_10010_M
  # folds = [fold0, fold1, fold2, fold3, fold4]
  folds = [fold0, fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9]
  return folds

def _get_minnum(test_folds, fold_data):
  test_min_feat = fold_data.shape[0]
  for test_person in test_folds:
    test_person_num = fold_data[fold_data['Person']==test_person].shape[0]
    if test_person_num < test_min_feat:
      test_min_feat = test_person_num
  return test_min_feat

def data_split(folds, df_data):
  train_list, test_list = [], []
  for test_folds in folds:
    test_pd = pd.DataFrame()
    train_pd = df_data.copy()
    fold_data = df_data.copy()
    for test_person in test_folds:
      person_data = fold_data[fold_data['Person']==test_person]
      # print(test_person, person_data.shape)
      test_pd = pd.concat([test_pd, person_data])
      drop_index = train_pd[train_pd['Person']==test_person].index
      train_pd = train_pd.drop(drop_index)
    train_list.append(train_pd.reset_index(drop=True))
    test_list.append(test_pd.reset_index(drop=True))
  print('data split done')
  return train_list, test_list