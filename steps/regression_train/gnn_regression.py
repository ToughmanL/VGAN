#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 nn_regression.py
* @Time 	:	 2023/03/23
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 graph neural network for regression
'''

import os
import yaml
import json
import random
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import metrics
import torch

from utils.normalization import Normalization
from utils.file_process import mkdir, dataframe_downsample
from utils.train_test_split import normal_test_person, data_split
from utils.checkpoint import load_checkpoint, migration
from utils.feat_select import feat_sel, score2class, NullDelete
from utils.extract_args import get_args
from utils.accelerator import DataLoaderX
from utils.evaluation import regression_eval, classfication_eval
from utils.loss_fun import focal_loss
from utils.init_models import init_model

from local import cmvn
from local.dataset import GnnDataLoder, IterDataset
from local.trainer import NNTrainer, GATTrainer

import warnings
warnings.filterwarnings('ignore')

ddp = False
if ddp:
  import torch.multiprocessing as mp
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel as DDP
  dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))
  local_rank = dist.get_rank()
  torch.cuda.set_device(local_rank)
  device = torch.device('cuda', local_rank)
  # 固定随机种子
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

class AudioRegression():
  def __init__(self, args, test_flag):
    self.acu_feats = None
    self.NORM = Normalization()
    self.y_scaler = None
    self.test_flag = test_flag
    self.train_list = []
    self.test_list = []
    label_score = {'Frenchay':116, 'fanshe':12, 'huxi':8, 'chun':20, 'he':8, 'ruane':12, 'hou':16, 'she':24,'yanyu':16, 'ljri':52, 'rlvt':64}
    self.label_score = {key: label_score[key] for key in args.label if key in label_score}
    self.fold_list = [int(x) for x in args.fold]
    self.sample_rate = 0.1
    self.feat_type = ''
    self.result_dir = ''
    self.cmvn = {}
    self.feat_dict = {}
    with open(args.config, 'r') as fin:
      self.configs = yaml.load(fin, Loader=yaml.FullLoader)
    # 分类还是回归
    self.task_type = 'classfication' if 'classfication' in self.configs else 'regression'
    # 定制元音
    self.vowels = ['a','o','e','i','u','v'] if 'vowels' not in self.configs else list(self.configs['vowels'])
    # 多卡
    if not ddp:
      mkdir(self.configs['model_dir'] + '/models/')
      mkdir(self.configs['model_dir'] + '/pics/')
      self.device = torch.device('cuda:{}'.format(str(args.gpu)))
    else:
      self.device = device
    self.Middle_feat = False
    self.middle_feats = []

  def _normalization(self, feat_type, raw_acu_feats):
    norm_acu_feats = None
    if feat_type == 'loop_featnorm' or feat_type == 'gop_loop_featnorm' or feat_type == 'loop_v_featnorm' or feat_type == 'loop_av_featnorm' or feat_type == 'gop' or feat_type == 'Liuartifeat': # 为所有特征计算标准差
      norm_acu_feats = self.NORM.class_normalization(raw_acu_feats, 1, -9) # 91 -9
    elif feat_type == 'loop_featnorm_xy':
      norm_acu_feats, self.y_scaler = self.NORM.class_normalization_x_y(raw_acu_feats, 1, -9)
    elif feat_type == 'loop_comsyll': # 为common特征计算音节标准差，为artic特征计算特征标准差
      norm_acu_feats = self.NORM.half_normalization(raw_acu_feats, 1, 8)
    else: # 排除mfcc和egemaps
      norm_acu_feats = self.NORM.class_normalization(raw_acu_feats, 1, -21) # 9个标签，6个MFCC, 6个egemaps
    # 填充、抽取
    # norm_acu_feats = self.NORM.fill_non(norm_acu_feats).dropna().reset_index(drop=True)
    norm_acu_feats = norm_acu_feats.dropna().reset_index(drop=True)
    return norm_acu_feats

  def _read_featdict(self, feat_path):
    with open(feat_path) as fp:
      feat_dict = json.load(fp)
    for key, value in feat_dict.items():
      feat_dict[key]['featdata'] = np.array(value['featdata']).astype(np.float32)
    return feat_dict

  def _read_feats(self, csv_path, feat_type, sample_rate):
    self.sample_rate = sample_rate
    self.feat_type = feat_type
    if 'cmvn_path' in self.configs:
      if self.configs['cmvn_path'] != 'None':
        self.cmvn = cmvn.read_cmvn_np(self.configs['cmvn_path'])[feat_type.replace('segment', '')]
    if 'feat_path' in self.configs:
      if self.configs['feat_path'] != 'None':
        self.feat_dict = self._read_featdict(self.configs['feat_path'])
    raw_acu_feats = pd.read_csv(csv_path)
    # 'ljri':52, 'rlvt':64
    # raw_acu_feats['ljri'] = raw_acu_feats['chun'] + raw_acu_feats['he'] + raw_acu_feats['huxi'] + raw_acu_feats['yanyu']
    # raw_acu_feats['rlvt'] = raw_acu_feats['fanshe'] + raw_acu_feats['ruane'] + raw_acu_feats['hou'] + raw_acu_feats['she']
    if 'featnorm' in feat_type or 'egemaps' == feat_type or 'gop' == feat_type or 'Liuartifeat' == feat_type or 'papi_cmlrv' == feat_type or 'phonation' == feat_type or 'articulation' == feat_type or 'prosody' == feat_type or 'papi' == feat_type or 'papi_cropavi' == feat_type or 'lip' == feat_type or 'papi_lip' == feat_type or 'papi_cmrlv' == feat_type or 'lip_cmlrv' == feat_type or feat_type == 'papi_lipcmrlv':
      raw_acu_feats = self._normalization(feat_type, raw_acu_feats)
      raw_acu_feats = feat_sel(feat_type, self.vowels, raw_acu_feats)
    elif 'segment' in feat_type:
      raw_acu_feats = feat_sel(feat_type, self.vowels, raw_acu_feats)
    else: # 剩下的很多特征暂且使用mfcc的地址，在dataprocessing中会改地址
      raw_acu_feats = feat_sel('mfcc', self.vowels, raw_acu_feats)
    feat_dir = self.configs.get('feat_dir', 'data/segment_data/')
    raw_acu_feats = NullDelete(feat_type, raw_acu_feats, feat_dir)
    # self.acu_feats = raw_acu_feats.sample(frac=sample_rate, random_state=1).reset_index(drop=True) # 此处已经shuffle了
    self.acu_feats = dataframe_downsample(raw_acu_feats, sample_rate)
    if self.task_type == 'classfication':
      self.acu_feats = score2class('Frenchay', self.acu_feats)
    print('Read feats finised')

  def _plot_regressor(self, fname, ref, hyp):
    # plt.figure()
    plt.plot(np.arange(len(ref)), ref,'go-',label='true value')
    plt.plot(np.arange(len(ref)),hyp,'ro-',label='predict value')
    plt.title(os.path.basename(fname).split('.')[0])
    plt.legend()
    # plt.show()
    plt.savefig(fname, dpi=120, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches='tight', pad_inches=0.2, frameon=None, metadata=None)
    plt.clf()

  def _save_result(self, data, label):
    model_path = self.configs['model_dir'] + label + '_value.csv'
    data.to_csv(model_path, index=False)

  def _train_test_split(self):
    folds = normal_test_person()
    self.train_list, self.test_list = data_split(folds, self.acu_feats)

  def _pad_batch(self, data, batch_size):
    pad_size = batch_size - (len(data) % batch_size)
    pad_data = data.sample(n=pad_size, random_state=1)
    return pd.concat([data, pad_data]).reset_index(drop=True)

  def _load_data(self, nodemode, train, test, val, label):
    BatchSize = self.configs['BATCH_SIZE']
    val_loader, test_loader, train_loader = None, None, None
    feat_dir = self.configs.get('feat_dir', "/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/data/segment_data")
    if self.configs['dataloader'] == 'geo_dataloader':
      GD = GeometricData(nodemode, label)
      if not self.test_flag:
        train_dataset = GD.get_graph_data(train)
        val_dataset = GD.get_graph_data(val)
        val_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=BatchSize, shuffle=True)
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
      test_dataset = GD.get_graph_data(test)
      test_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=BatchSize, shuffle=False)
    elif self.configs['dataloader'] == 'gnn_dataloader':
      test_dataset = GnnDataLoder(test, self.feat_type, self.feat_dict, self.cmvn, nodemode, label)
      test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=16)
      if not self.test_flag:
        # val_dataset = GnnDataLoder(val, self.feat_type, self.feat_dict, self.cmvn, nodemode, label)
        train_dataset = GnnDataLoder(train, self.feat_type, self.feat_dict, self.cmvn, nodemode, label)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BatchSize, shuffle=True, num_workers=16)
        val_loader = test_loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=16)
    elif self.configs['dataloader'] == 'IterDataset':
      # torch.utils.data.DataLoader -> DataLoaderX
      test_loader = DataLoaderX(IterDataset(test, self.feat_type, label, feat_dir), batch_size=BatchSize, num_workers=16, drop_last=True)
      if not self.test_flag:
        # val_loader = DataLoaderX(IterDataset(val, self.feat_type, label), batch_size=BatchSize, num_workers=16, drop_last=True)
        val_loader = test_loader
        train_loader = DataLoaderX(IterDataset(train, self.feat_type, label, feat_dir), batch_size=BatchSize, num_workers=16, drop_last=True)
    return val_loader, test_loader, train_loader

  def _model_train(self, dmodel, fold_i):
    label = dmodel['label']
    name = dmodel['name']
    nodemode = self.configs['nodemode']
    BatchSize = self.configs['BATCH_SIZE']
    train, test = self.train_list[fold_i], self.test_list[fold_i]
    train = self._pad_batch(train, BatchSize)
    test = self._pad_batch(test, BatchSize)
    val = test.copy()
    Y_test = test[label]

    # test_new = test.copy()
    # test_new.drop(columns=['a_mfcc', 'o_mfcc','e_mfcc', 'i_mfcc','u_mfcc', 'v_mfcc'], inplace=True)
    # self.feat_type = 'loop_featnorm'
    model = init_model(self.configs, name, label, fold_i)
    val_loader, test_loader, train_loader = self._load_data(nodemode, train, test, val, label)


    model.to(self.device)
    print(model)
    if ddp:
      model = DDP(model, device_ids=[self.device], find_unused_parameters=True)

    epoch = self.configs['EPOCHS']
    if self.task_type == 'classfication':
      # loss_function = torch.nn.CrossEntropyLoss()
      # loss_function = torch.nn.BCEWithLogitsLoss()
      loss_function = focal_loss(device=self.device)
    elif self.task_type == 'regression':
      loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['LEARNING_RATE'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)
    if name == 'GAT':
      Trainer = GATTrainer(self.configs, label, model, loss_function, optimizer, scheduler)
    else:
      Trainer = NNTrainer(self.configs, label, model, loss_function, optimizer, scheduler)

    if not self.test_flag:
      Trainer.train(train_loader, val_loader, self.device, fold_i)

    model_path = self.configs['model_dir'] + '/models/' + label + '_fold_' + str(fold_i) + '_best.pt'
    config = load_checkpoint(model, model_path)

    if self.Middle_feat:
      # 获取中间值
      middle_value = Trainer.get_middle_layer(test_loader, 'linear3', self.device)
      self.middle_feats.append(middle_value)
    # 测试
    hpy_list, ref_list = Trainer.evaluate(test_loader, self.device)

    X_test_predict = np.vstack(hpy_list)
    X_test_target = np.vstack(ref_list)
    X_test_predict = X_test_predict.reshape(-1, 1)
    X_test_target = X_test_predict.reshape(-1, 1)

    if self.configs['feat_type'] == 'loop_featnorm_xy':
      X_test_predict = X_test_predict + self.y_scaler[label]['mean']
      X_test_target = X_test_target  + self.y_scaler[label]['mean']
      Y_test = Y_test + self.y_scaler[label]['mean']

    # 添加了这里
    X_test_predict[X_test_predict < 0] = 0
    X_test_predict[X_test_predict > self.label_score[label]] = self.label_score[label]

    if len(test) > len(X_test_predict):
      test = test.iloc[:len(X_test_predict),:]
      Y_test = Y_test.iloc[:len(X_test_predict)]
    sylla_new = pd.DataFrame(test[label], columns=[label])
    sylla_new = pd.concat([sylla_new, pd.DataFrame(X_test_predict, columns=['Predict'])], axis=1)
    if self.test_flag:
      # syllabel_test = pd.concat([test, sylla_new], axis=1)
      syllabel_test = pd.concat([test.iloc[:,0], sylla_new], axis=1)
    else:
      syllabel_test = pd.concat([test.iloc[:,0], sylla_new], axis=1)

    person_test = pd.DataFrame()
    # Subject level (test)
    for person in test['Person'].unique():
      person_index = test['Person'].isin([person])
      person_tmp = pd.DataFrame({'Person': person, label: np.mean(Y_test[person_index]), 'Predict': np.mean(X_test_predict[person_index])}, index=[1])
      person_test = person_test.append(person_tmp, ignore_index=True)

    if not self.test_flag:
      result_info = {'name':name, 'label':label, 'foild':fold_i, 'countlevel':'syllable'}
      if self.task_type == 'regression':
        result_info.update(regression_eval(syllabel_test[label], syllabel_test['Predict']))
      elif self.task_type == 'classfication':
        result_info.update(classfication_eval(syllabel_test[label], syllabel_test['Predict']))
      print(result_info)
    return syllabel_test, person_test
  
  def _corss_vali(self, model_label):
    label = model_label['label']
    name = model_label['name']
    person_pred_total = pd.DataFrame()
    syllabel_pred_total = pd.DataFrame()
    for fold_i in self.fold_list:
      syllabel_test, person_test = self._model_train(model_label, fold_i)
      person_pred_total = pd.concat([person_pred_total, person_test], axis=0)
      syllabel_pred_total = pd.concat([syllabel_pred_total, syllabel_test], axis=0)

    if self.test_flag:
      self._save_result(syllabel_pred_total, label)
    syllabel_pred_total = syllabel_pred_total.sort_values(by=['Person']).reset_index(drop=True)
    person_pred_total = person_pred_total.sort_values(by=['Person']).reset_index(drop=True)

    syll_result_info = {'name':name, 'label':label, 'foild':fold_i, 'countlevel':'syllable'}
    person_result_info = {'name':name, 'label':label, 'foild':fold_i, 'countlevel':'person'}
    if self.task_type == 'regression':
      syll_result_info.update(regression_eval(syllabel_pred_total[label], syllabel_pred_total['Predict']))
      person_result_info.update(regression_eval(person_pred_total[label], person_pred_total['Predict']))
    elif self.task_type == 'classfication':
      syll_result_info.update(classfication_eval(syllabel_pred_total[label], syllabel_pred_total['Predict']))
      person_result_info.update(classfication_eval(person_pred_total[label].round(0).astype(np.int32), person_pred_total['Predict'].round(0).astype(np.int32)))
    print(syll_result_info)
    print(person_result_info)

    # 保存result信息
    self._plot_regressor(self.result_dir + '/Person' + label + '_' + name + '.png', person_pred_total[label], person_pred_total['Predict'])
    self._plot_regressor(self.result_dir + '/syllabel' + label + '_' + name + '.png', syllabel_pred_total[label], syllabel_pred_total['Predict'])
    return person_result_info

  def compute_result(self):
    csv_path = self.configs['feat_csv']
    feat_type = self.configs['feat_type']
    sample_rate = self.configs['sample_rate']
    self._read_feats(csv_path, feat_type, sample_rate)
    self.result_dir = '{}/{}'.format(self.configs['model_dir'], self.feat_type+'_'+str(self.sample_rate))
    mkdir(self.result_dir)
    self._train_test_split()

    paras_list = []
    results = []
    for label in self.label_score.keys():
      print(label)
      dmodel = {'label':label}
      dmodel.update({'name':self.configs['model']})
      results.append(self._corss_vali(dmodel))
    
    if self.Middle_feat:
      merged_df = pd.concat(self.middle_feats, ignore_index=True)
      csv_path = '{}/middle_{}_{}_{}.csv'.format(self.configs['model_dir'] ,self.configs['model'], feat_type, str(sample_rate))
      merged_df.to_csv(csv_path)

    df_result = pd.DataFrame(results)
    csv_path = '{}/acoustic_{}_{}_{}.csv'.format(self.configs['model_dir'] ,self.configs['model'], feat_type, str(sample_rate))
    df_result.to_csv(csv_path, mode='a')

if __name__ == "__main__":
  args = get_args()
  print(args)
  AR = AudioRegression(args, args.test_flag)
  AR.compute_result()

# python gnn_regression.py --config conf/train_dropdnn.yaml --gpu 0 --label Frenchay --test_flag --fold 0 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=25641 gnn_regression.py

