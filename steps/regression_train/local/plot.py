#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 plot.py
* @Time 	:	 2023/02/23
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import seaborn as sns
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib
import matplotlib.pyplot as plt
from utils.normalization import Normalization

def plot_regressor(fname, ref, hyp):
  # plt.figure()
  plt.plot(np.arange(len(ref)), ref,'go-',label='true value')
  plt.plot(np.arange(len(ref)),hyp,'ro-',label='predict value')
  plt.title(os.path.basename(fname).split('.')[0])
  plt.legend()
  # plt.show()
  plt.savefig(fname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
  plt.clf()

def plot_importance(importances, train, model_label, result_dir):
  train_label = train.iloc[:,1:-9]
  impo_sum = importances[0]
  for i in range(1, len(importances)):
    impo_sum += importances[i]
  impo_mean = impo_sum/len(importances)
  indices = np.argsort(impo_mean)
  indices = indices[-20:]

  plt.figure(figsize=(18, 50), dpi=400)
  fig, ax = plt.subplots()
  ax.barh(range(len(indices)), impo_mean[indices])
  ax.set_yticks(range(len(indices)))
  _ = ax.set_yticklabels(np.array(train_label.columns)[indices],fontsize=6)
  plt.savefig(result_dir + '/{}_gbdt.png'.format(model_label), dpi=400, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
  plt.clf()

def polt_feature_relationship(raw_data):
  # 画图看特征间的关系,主要是变量两两之间的关系（线性或非线性，有无明显较为相关关系）
  # matplotlib.use('TkAgg')
  NORM = Normalization()
  norm_acu_feats = NORM.class_normalization(raw_data, 2, -9)

  feature_name = norm_acu_feats.columns[1:-8]
  df_data = norm_acu_feats[feature_name].dropna().sample(n=2000, random_state=0).reset_index(drop=True)

  plt.figure(figsize=(96, 96))
  sns.pairplot(df_data.iloc[:,:-1], kind="reg",diag_kind="kde")
  plt.savefig('ContentRelevance.png')
  plt.clf()
  
  plt.figure(figsize=(96, 96))
  sns.pairplot(df_data, hue='Frenchay')
  plt.savefig('CategoryCorrelation.png')
  plt.clf()

  # # 相关性分析
  # plt.figure(figsize=(96, 96))
  # corr_data = df_data.iloc[:,:-1].corr()
  # corr_data = np.asarray(corr_data)
  # sns.heatmap(corr_data, xticklabels=feature_name, yticklabels=feature_name, annot=True, cbar=True)
  # plt.xticks(fontsize=30, rotation=90)
  # plt.yticks(fontsize=30)
  # plt.savefig('CorrelationAnalysis.png')
  # plt.clf()

  # # 特征箱型图
  # plt.figure(figsize=(96, 96))
  # sns.boxplot(data=df_data.iloc[:, 0:-1])
  # plt.xticks(fontsize=30, rotation=90)
  # plt.yticks(fontsize=30)
  # plt.savefig('Box.png')
  # plt.clf()

def polt_values(value_path):
  file_path_dict = {}
  for root, dirs, files in os.walk(value_path):
    for file in files:
      name_list = file.split('_')
      if 'value.csv' in name_list:
        file_path_dict[name_list[0]] = os.path.join(root, file)
  
  for label, path in file_path_dict.items():
    data = pd.read_csv(path)
    for person in data['Person'].unique():
      person_data = data.loc[data['Person'] == person]
      ref_data = person_data[[label]]
      phy_data = person_data[['Predict']]
      plt.hist(ref_data, bins=30, color="blue", edgecolor="black", alpha=0.7)
      plt.hist(phy_data, bins=30, color="green", edgecolor="black", alpha=0.7)
      pic_path = path.replace('_value.csv', '_{}_hist.png'.format(person))
      plt.savefig(pic_path)
      plt.clf()

if __name__ == "__main__":
  # csv_path = '../data/result_intermediate/acoustic_loop_feats_0209.csv'
  # raw_acu_feats = pd.read_csv(csv_path)
  # polt_feature_relationship(raw_acu_feats)

  value_path = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/regression_train/models/nn_model/GAT1NN2_2'
  polt_values(value_path)