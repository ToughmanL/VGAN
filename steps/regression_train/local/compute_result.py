#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 CustomLayer.py
* @Time 	:	 2023/03/16
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import os
import pandas as pd
import numpy as np
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

class ComputeResult():
  def __init__(self):
    self.tasks = ['huxi', 'chun', 'he', 'ruane', 'hou', 'she','yanyu']
    self.task_data = {}
    self.model_path = 'models/nn_model/'
  
  def read_results(self, path):
    file_list = os.listdir(path)
    for file in file_list:
      if 'value.csv' in file:
        file_path = os.path.join(path, file)
        self.task_data[file.split('_')[0]] = pd.read_csv(file_path, index_col=0)
    if len(self.task_data.keys()) != 9:
      print('Missing result')
      return -1
    return 0
  
  def _mode_mean(self, data):
    bins = 10
    bin_wide = (data.max() - data.min())/bins
    best_data = np.arange(1)
    for i in range(bins):
      sele_data = np.where((data>=(i*bin_wide + data.min()))&(data<=((i+1)*bin_wide + data.min())), data, None)
      sele_data = sele_data[sele_data != None]
      if sele_data.size > best_data.size:
        best_data = sele_data
    return np.mean(best_data)

  def compute_sum_frenchay(self):
    sum_data = self.task_data['Frenchay'].copy()
    sum_data['Predict'] = 0
    for task in self.tasks:
      sum_data['Predict'] = sum_data['Predict'] + self.task_data[task]['Predict']
    return sum_data
  
  def syllabel_rmse_r2(self, name, label, syllabel_test):
    syllabel_test_r2 = metrics.r2_score(syllabel_test[label], syllabel_test['Predict'])
    syllabel_test_rmse = np.sqrt(metrics.mean_squared_error(syllabel_test[label], syllabel_test['Predict']))
    result_info = {'name':name, 'label':label, 'sylla_test_rmse':syllabel_test_rmse, 'sylla_test_r2':syllabel_test_r2}
    print(result_info)

  def person_mean_rmse_r2(self, name, label, syllabel_test):
    person_test = pd.DataFrame()
    for person in syllabel_test['Person'].unique():
      person_index = syllabel_test['Person'].isin([person])
      person_tmp = pd.DataFrame({'Person': person, label: np.mean(syllabel_test[person_index][label]), 'Predict': np.mean(syllabel_test[person_index]['Predict'])}, index=[1])
      person_test = person_test.append(person_tmp, ignore_index=True)
    person_test_r2 = metrics.r2_score(person_test[label], person_test['Predict'])
    person_test_rmse = np.sqrt(metrics.mean_squared_error(person_test[label], person_test['Predict']))
    result_info = {'name':name, 'label':label, 'person_test_rmse':person_test_rmse, 'person_test_r2':person_test_r2}
    print(result_info)
  
  def person_mode_rmse_r2(self, name, label, syllabel_test):
    person_test = pd.DataFrame()
    for person in syllabel_test['Person'].unique():
      person_index = syllabel_test['Person'].isin([person])
      person_tmp = pd.DataFrame({'Person': person, label: np.mean(syllabel_test[person_index][label]), 'Predict': self._mode_mean(syllabel_test[person_index]['Predict'])}, index=[1])
      person_test = person_test.append(person_tmp, ignore_index=True)
    person_test_r2 = metrics.r2_score(person_test[label], person_test['Predict'])
    person_test_rmse = np.sqrt(metrics.mean_squared_error(person_test[label], person_test['Predict']))
    result_info = {'name':name, 'label':label, 'person_test_rmse':person_test_rmse, 'person_test_r2':person_test_r2}
    print(result_info)
  
  def get_all_models(self):
    model_names = os.listdir(self.model_path)
    tasks = ['Frenchay']
    tasks = tasks + self.tasks
    for name in model_names:
      name_path = self.model_path + name
      if self.read_results(name_path) != 0:
        continue
      sum_data = self.compute_sum_frenchay()
      # self.syllabel_rmse_r2(name, 'Frenchay', sum_data)
      for task in tasks:
        self.syllabel_rmse_r2(name, task, self.task_data[task])
        self.person_mean_rmse_r2(name, task, self.task_data[task])
        self.person_mode_rmse_r2(name, task, self.task_data[task])
      

if __name__ == "__main__":
  CR = ComputeResult()
  CR.get_all_models()