#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 evaluation.py
* @Time 	:	 2023/12/25
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import numpy as np
from sklearn import metrics

def regression_eval(label, predict):
  syllabel_test_r2 = metrics.r2_score(label, predict)
  syllabel_test_rmse = np.sqrt(metrics.mean_squared_error(label, predict))
  result_info = {'rmse':syllabel_test_rmse, 'r2':syllabel_test_r2}
  return result_info

def classfication_eval(label, predict):
  acc = metrics.accuracy_score(label, predict)
  confusion_matrix = metrics.confusion_matrix(label, predict, labels=[0,1,2,3])
  precision_0 = metrics.precision_score(label, predict, pos_label=0, average='micro')
  precision_1 = metrics.precision_score(label, predict, pos_label=1, average='micro')
  precision_2 = metrics.precision_score(label, predict, pos_label=2, average='micro')
  precision_3 = metrics.precision_score(label, predict, pos_label=3, average='micro')
  recall_0 = metrics.recall_score(label, predict, pos_label=0, average='micro')
  recall_1 = metrics.recall_score(label, predict, pos_label=1, average='micro')
  recall_2 = metrics.recall_score(label, predict, pos_label=2, average='micro')
  recall_3 = metrics.recall_score(label, predict, pos_label=3, average='micro')
  f1_score = metrics.f1_score(label, predict, average='micro')
  result_info = {'acc':acc, 'precision_0':precision_0, 'precision_1':precision_1, 'precision_2':precision_2, 'precision_3':precision_3, 'recall_1':recall_1, 'recall_2':recall_2, 'recall_3':recall_3, 'f1_score':f1_score}
  return result_info