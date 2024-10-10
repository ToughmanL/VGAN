#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 a_regression.py
* @Time 	:	 2022/12/05
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import os
import shutil
import yaml
import pandas as pd
import numpy as np
from multiprocessing import Process
from matplotlib import pyplot as plt

import joblib
from sklearn import linear_model, svm, neural_network
from sklearn import model_selection, preprocessing, metrics, base, decomposition
from sklearn import preprocessing

from utils.multi_process import MultiProcess
from utils.regress_models import RegressModel
from utils.normalization import Normalization
from utils.feat_select import feat_sel
from utils.file_process import mkdir, dataframe_downsample
from utils.train_test_split import lsj_test_person, normal_test_person, data_split

import warnings
warnings.filterwarnings('ignore')

class AudioRegression():
  def __init__(self, config_path, train_flag=False):
    self.acu_feats = None
    self.NORM = Normalization()
    self.train_list = []
    self.test_list = []
    # self.label_score = {'Frenchay':116,'fanshe':12, 'huxi':8, 'chun':20, 'he':8, 'ruane':12, 'hou':16, 'she':24,'yanyu':16}
    self.label_score = {'Frenchay':116}
    self.result_dir = ''
    self.train_flag = train_flag
    with open(config_path, 'r') as fin:
      self.configs = yaml.load(fin, Loader=yaml.FullLoader)
    if 'vowels' in self.configs:
      self.vowels = list(self.configs['vowels'])
    else:
      self.vowels = ['a','o','e','i','u','v']
    self.model_dir = self.configs['model_dir']
    self.sample_rate = self.configs['sample_rate']
    self.feat_type = self.configs['feat_type']
    self.csv_path = self.configs['feat_csv']
    self.multi_num = self.configs['multi_num']
    mkdir(self.model_dir)

  def _normalization(self, feat_type, raw_acu_feats):
    if feat_type == 'loop_featnorm' or feat_type == 'loop_featnorm_gop84': # 为所有特征计算标准差
      norm_acu_feats = self.NORM.class_normalization(raw_acu_feats, 1, -9)
    elif feat_type == 'loop_comsyll': # 为common特征计算音节标准差，为artic特征计算特征标准差
      norm_acu_feats = self.NORM.half_normalization(raw_acu_feats, 2, 8)
    elif feat_type == 'orilsj_featnorm': # 为所有特征计算标准差
      raw_acu_feats = raw_acu_feats.drop(['syllabel_acc', 'total_acc'], axis=1)
      norm_acu_feats = self.NORM.class_normalization(raw_acu_feats, 3,-9) # 鲁尚军特征标准化
    elif feat_type == 'lsj_featnorm': # 为所有特征计算标准差
      drop_acu_feats = raw_acu_feats.drop(['syllabel_acc', 'total_acc'], axis=1)
      norm_acu_feats = self.NORM.class_normalization(drop_acu_feats, 3,-9)
    elif feat_type == 'lsj_sylla': # 为所有特征计算标准差
      drop_acu_feats = raw_acu_feats.drop(['syllabel_acc', 'total_acc'], axis=1)
      norm_acu_feats = self.NORM.lsj_syllable(drop_acu_feats, 3,-9)
    else:
      print('No such feat type')
      exit(-1)
    # 填充、抽取
    # norm_data = norm_acu_feats.fillna(0)
    norm_data = self.NORM.fill_non(norm_acu_feats).dropna().reset_index(drop=True)
    return norm_data

  def _read_feats(self, feat_type):
    raw_acu_feats = pd.read_csv(self.csv_path)
    if 'featnorm' in feat_type or 'egemaps' == feat_type or 'gop' == feat_type or 'Liuartifeat' == feat_type or 'papi_cmlrv' == feat_type:
      raw_acu_feats = feat_sel(feat_type, self.vowels, raw_acu_feats)
    else:
      raw_acu_feats = feat_sel('mfcc', self.vowels, raw_acu_feats)
    raw_acu_feats = self._normalization(feat_type, raw_acu_feats)
    # self.acu_feats = norm_data.sample(frac=self.sample_rate, random_state=1).reset_index(drop=True)
    self.acu_feats = dataframe_downsample(raw_acu_feats, self.sample_rate)
    print('read feats finised')

  def plot_regressor(self, fname, ref, hyp):
    # plt.figure()
    plt.plot(np.arange(len(ref)), ref,'go-',label='true value')
    plt.plot(np.arange(len(ref)),hyp,'ro-',label='predict value')
    plt.title(os.path.basename(fname).split('.')[0])
    plt.legend()
    # plt.show()
    plt.savefig(fname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
    plt.clf()

  def plot_importance(self, importances, train, model_label):
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
    plt.savefig(self.result_dir + '/{}_gbdt.png'.format(model_label), dpi=400, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
    plt.clf()

  def _save_result(self, data, label):
    model_path = self.configs['model_dir'] + '/' + label + '_value.csv'
    data.to_csv(model_path)

  def _train_test_split(self):
    folds = normal_test_person()
    self.train_list, self.test_list = data_split(folds, self.acu_feats)

  def _model_train(self, dmodel, fold_i):
    label = dmodel['label']
    train, test = self.train_list[fold_i], self.test_list[fold_i]
    X_train = train.iloc[:,:-9]
    Y_train = train[label]
    X_test = test.iloc[:,:-9]
    Y_test = test[label]
    name = dmodel['name']
    model = dmodel['model']

    model_path = self.model_dir + "{}_{}_fold{}.pkl".format(label, name, str(fold_i))
    
    person_test = pd.DataFrame()
    fitted_model = base.clone(model)
    if self.train_flag:
      if os.path.exists(model_path):
        fitted_model = joblib.load(model_path)
      else:
        fitted_model.fit(X_train.iloc[:,1:], Y_train)
        joblib.dump(fitted_model, model_path)
    else:
      if os.path.exists(model_path):
        fitted_model = joblib.load(model_path)
      else:
        print('no saved model')
        exit(-1)

    X_test_predict = fitted_model.predict(X_test.iloc[:, 1:])
    
    if 'gbdt' in name:
      importances = fitted_model.feature_importances_
    else:
      importances = None

    # 添加了这里
    X_test_predict[X_test_predict < 0] = 0
    X_test_predict[X_test_predict > self.label_score[label]] = self.label_score[label]

    sylla_new = pd.DataFrame(Y_test, columns=[label])
    sylla_new = pd.concat([sylla_new, pd.DataFrame(X_test_predict, columns=['Predict'])], axis=1)
    syllabel_test = pd.concat([X_test.iloc[:,0], sylla_new], axis=1)

    # Subject level (test)
    for person in X_test['Person'].unique():
      person_index = X_test['Person'].isin([person])
      person_tmp = pd.DataFrame({'Person': person, label: np.mean(Y_test[person_index]), 'Predict': np.mean(X_test_predict[person_index])}, index=[1])
      person_test = person_test.append(person_tmp, ignore_index=True)
    
    if self.train_flag:
      syllabel_test_r2 = metrics.r2_score(syllabel_test[label], syllabel_test['Predict'])
      syllabel_test_rmse = np.sqrt(metrics.mean_squared_error(syllabel_test[label], syllabel_test['Predict']))
      result_info = {'name':name, 'label':label, 'foild':fold_i, 'sylla_test_rmse':syllabel_test_rmse, 'sylla_test_r2':syllabel_test_r2}
      print(result_info)
    return syllabel_test, person_test, importances

  def _corss_vali(self, model_label):
    label = model_label['label']
    name = model_label['name']
    person_pred_total = pd.DataFrame()
    syllabel_pred_total = pd.DataFrame()
    fold_nums = len(self.test_list)
    importances_fold = []
    for fold_i in range(1, fold_nums):
      syllabel_test, person_test, importances = self._model_train(model_label, fold_i)
      person_pred_total = pd.concat([person_pred_total, person_test], axis=0)
      syllabel_pred_total = pd.concat([syllabel_pred_total, syllabel_test], axis=0)
      importances_fold.append(importances)
    if syllabel_pred_total.empty:
      return {'name':name, 'label':label,'sylla_test_rmse':None, 'sylla_test_r2':None, 'person_test_rmse':None, 'person_test_r2':None}

    self._save_result(syllabel_pred_total, label)
    syllabel_pred_total = syllabel_pred_total.sort_values(by=['Person']).reset_index(drop=True)
    person_pred_total = person_pred_total.sort_values(by=['Person']).reset_index(drop=True)
    if 'gbdt' in model_label['name']:
      self.plot_importance(importances_fold, self.train_list[0], label)

    syllabel_test_r2 = metrics.r2_score(syllabel_pred_total[label], syllabel_pred_total['Predict'])
    syllabel_test_rmse = np.sqrt(metrics.mean_squared_error(syllabel_pred_total[label], syllabel_pred_total['Predict']))
    person_test_r2 = metrics.r2_score(person_pred_total[label], person_pred_total['Predict'])
    person_test_rmse = np.sqrt(metrics.mean_squared_error(person_pred_total[label], person_pred_total['Predict']))

    name = model_label['name']
    result_info = {'name':name, 'label':label,'sylla_test_rmse':syllabel_test_rmse, 'sylla_test_r2':syllabel_test_r2, 'person_test_rmse':person_test_rmse, 'person_test_r2':person_test_r2}
    print(result_info)

    # 保存result信息
    self.plot_regressor(self.result_dir + '/Person' + label + '_' + name + '.png', person_pred_total[label], person_pred_total['Predict'])
    self.plot_regressor(self.result_dir + '/syllabel' + label + '_' + name + '.png', syllabel_pred_total[label], syllabel_pred_total['Predict'])
    return result_info

  def compute_result(self):
    self._read_feats(self.feat_type)
    self.result_dir = 'tmp/{}'.format(self.feat_type+'_'+str(self.sample_rate))
    mkdir(self.result_dir)
    self._train_test_split()

    paras_list = []
    results = []
    RM = RegressModel()
    models = RM.get_all_models()
    for model in models:
      for label in self.label_score.keys():
        dmodel = {'label':label}
        dmodel.update(model)
        paras_list.append(dmodel)
    if self.multi_num == 1:
      for para in paras_list:
        results.append(self._corss_vali(para))
    else:
      MP = MultiProcess()
      results = MP.multi_with_result(func=self._corss_vali, \
          arg_list=paras_list, process_num=self.multi_num)

    df_result = pd.DataFrame(results)
    df_result.to_csv('tmp/acoustic_{}_{}.csv'.format(self.feat_type, str(self.sample_rate)))
    print('regression finished')

if __name__ == "__main__":
  config = 'conf/train_ml.yaml'
  print(config)
  AR = AudioRegression(config, train_flag=True)
  AR.compute_result()
