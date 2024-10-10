#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 dataset.py
* @Time 	:	 2023/02/23
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
import torch.distributed as dist
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from torch_geometric.data import Data
from utils.multi_process import MultiProcess
from torch.utils.data import IterableDataset
from utils.data_processor import DataProcessor
from utils.feat_dimension_reduction import FeatDimReduction
from utils.accelerator import DataLoaderX


class NNDataLoder(torch.utils.data.Dataset):
  def __init__(self, X, y, scale_data=False):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.tensor(X.values.astype(np.float32))
      self.y = torch.tensor(y.values.astype(np.float32))

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

class BaseFeatDataLoder(torch.utils.data.Dataset):
  def __init__(self, feats, feat_type, feat_dict, cmvn, label):
    self.feat_type = feat_type
    self.targetlength = cmvn['targetlength']
    self.cmvn = cmvn
    self.feat_dict = feat_dict
    self.label = label
    feat_list = []
    row_dict_list = feats.T.to_dict().values()
    process_num = 1 if self.feat_type == 'stft' else 60
    feat_list, labels_list = self._feat_process(row_dict_list, process_num)
    # if self.feat_type == 'egemaps':
    #   FDR = FeatDimReduction()
    #   feat_list=FDR.PCAFeatDimRe(feat_list)
    self.X = torch.FloatTensor(feat_list).unsqueeze(dim=1)
    self.y = torch.FloatTensor(labels_list)
    random.seed(0)

  def _get_feat(self, dataframe_dict):
    feat_np_list = []
    for col_name, feat_path in dataframe_dict.items():
      if self.feat_type not in col_name:
        continue # 跳过无关特征
      np_data = self.feat_dict[feat_path.split('.')[0]]['featdata']
      if self.feat_type == 'mfcc':
        if np_data.shape[1] > self.targetlength:
          np_data = np_data[:,:self.targetlength]
        elif np_data.shape[1] < self.targetlength:
          np_data = np.pad(np_data, ((0, 0), (0, self.targetlength-np_data.shape[1])))
        np_data = (np_data - np.expand_dims(self.cmvn['mean'], axis=1))
        np_data = (np_data * np.expand_dims(self.cmvn['var'], axis=1))[0:13, :]
      elif self.feat_type == 'egemaps':
        np_data = np_data - self.cmvn['mean']
        np_data = np_data * self.cmvn['var']
      feat_np_list.append(np_data)
    if len(feat_np_list) == 1:
      feat_np_arr = feat_np_list[0]
    else:
      feat_np_arr = np.hstack(feat_np_list)
    return [feat_np_arr, float(dataframe_dict[self.label])]

  def _feat_process(self, row_dict_list, process_num=60):
    data_list = []
    if process_num == 1:
      for row_dict in row_dict_list:
        data_list.append(self._get_feat(row_dict))
    else:
      MP = MultiProcess()
      data_list = MP.multi_with_result(func=self._get_feat, arg_list=row_dict_list, process_num=process_num)
    feat_list = [data_list[i][0] for i in range(len(data_list))]
    labels_list = [data_list[i][1] for i in range(len(data_list))]
    return feat_list, labels_list

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    return self.X[i], self.y[i]

class GnnDataLoder(torch.utils.data.Dataset):
  def __init__(self, feats, feat_type, feat_dict, cmvn, nodemode, label):
    # config
    self.feat_type = feat_type
    self.cmvn = cmvn
    if 'targetlength' in cmvn:
      self.targetlength = cmvn['targetlength']
    self.feat_dict = feat_dict
    self.node_label = []
    self.node_dict = {}
    self.label = label
    self.arti_feats = ['tougue_dist', 'jaw_dist', 'move_degree', 'VSA', 'VAI', 'FCR']
    if nodemode == 'vowel':
      self.node_label = ['u', 'i', 'a', 'e', 'o', 'v']
    elif nodemode == 'commonfeat':
      self.node_label = ['Jitter', 'Shimmer', 'HNR', 'gne', 'vfer', 'F1_sd', 'F2_sd', 'F3_sd', 'Intensity_mean', 'Intensity_sd', 'Vowel_dur', 'Syllable_dur','gop_con','gop_vow']
    elif nodemode == 'allfeat':
      self.node_label = feats.columns[:-9]
    for i in range(len(self.node_label)):
      self.node_dict[self.node_label[i]] = i
    self.node_num = len(self.node_label)
    feat_list, labels_list = self._get_graph_data(feats)
    # if self.feat_type == 'egemaps':
    #   FDR = FeatDimReduction()
    #   feat_list=FDR.PCAFeatDimRe(feat_list)
    self.data_tensor = torch.FloatTensor(feat_list)
    self.label_tensor = torch.FloatTensor(labels_list)
    print("dataset prepared")
  
  def _get_graph(self, row_dict):
    features = []
    for nl in self.node_label:
      feat_vec, arti_feat_value = [], []
      for key, value in row_dict.items():
        if nl == key.split('-')[0]:
          feat_vec.append(value)
        if key in self.arti_feats:
          arti_feat_value.append(value)
      feat_vec.extend(arti_feat_value) # 在最后一层加上arti特征
      features.append(feat_vec)
    return [features, float(row_dict[self.label])]
  
  def _feat_norm(self, dataframe_dict):
    feat_np_list = []
    for col_name, feat_path in dataframe_dict.items():
      if self.feat_type not in col_name: # 过滤掉不相关特征
        continue
      np_data = self.feat_dict[feat_path.split('.')[0]]['featdata']
      if self.feat_type == 'mfcc':
        if np_data.shape[1] > 80:
          np_data = np_data[:,:80]
        elif np_data.shape[1] < 80:
          np_data = np.pad(np_data, ((0,0),(0,self.targetlength-np_data.shape[1])),'constant',constant_values = (0,0))
        np_data = (np_data - self.cmvn['mean'][:, np.newaxis])
        np_data = (np_data * self.cmvn['var'][:, np.newaxis])[0:13, :]
      elif self.feat_type == 'egemaps':
        np_data = np_data - self.cmvn['mean']
        np_data = np_data * self.cmvn['var']
      feat_np_list.append(np_data)
    return [feat_np_list, float(dataframe_dict[self.label])]

  def _get_graph_data(self, feat_data, process_num=60):
    data_list = []
    row_dict_list = feat_data.T.to_dict().values()
    MP = MultiProcess()
    if process_num > 1:
      if self.feat_type == 'loop_featnorm' or self.feat_type == 'gop_loop_featnorm' or self.feat_type == 'loop_av_featnorm' or self.feat_type == 'loop_v_featnorm' or self.feat_type == 'gop' or self.feat_type == 'Liuartifeat':
        data_list = MP.multi_with_result(func=self._get_graph, arg_list=row_dict_list, process_num=process_num)
      elif self.feat_type == 'mfcc' or self.feat_type == 'egemaps':
        data_list = MP.multi_with_result(func=self._feat_norm, arg_list=row_dict_list, process_num=process_num)
    elif process_num == 1:
      if self.feat_type == 'loop_featnorm' or self.feat_type == 'gop_loop_featnorm' or self.feat_type == 'loop_v_featnorm'or self.feat_type == 'loop_av_featnorm' or feat_type == 'gop' or feat_type == 'Liuartifeat':
        for row_dict in row_dict_list:
          data_list.append(self._get_graph(row_dict))
      elif self.feat_type == 'mfcc' or self.feat_type == 'egemaps':
        for row_dict in row_dict_list:
          data_list.append(self._feat_norm(row_dict))
    feat_list = [data_list[i][0] for i in range(len(data_list))]
    labels_list = [data_list[i][1] for i in range(len(data_list))]
    return feat_list, labels_list

  def __len__(self):
      return len(self.data_tensor)

  def __getitem__(self, i):
      return self.data_tensor[i], self.label_tensor[i]

class Processor(IterableDataset):
  def __init__(self, source, f, *args, **kw):
      assert callable(f)
      self.source = source
      self.f = f
      self.args = args
      self.kw = kw

  def set_epoch(self, epoch):
      self.source.set_epoch(epoch)

  def __iter__(self):
      """ Return an iterator over the source dataset processed by the
        given processor.
      """
      assert self.source is not None
      assert callable(self.f)
      return self.f(iter(self.source), *self.args, **self.kw)

  def apply(self, f):
      assert callable(f)
      return Processor(self, f, *self.args, **self.kw)

class DistributedSampler:
  def __init__(self, shuffle=True, partition=True):
      self.epoch = -1
      self.update()
      self.shuffle = shuffle
      self.partition = partition

  def update(self):
      assert dist.is_available()
      if dist.is_initialized():
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
      else:
        self.rank = 0
        self.world_size = 1
      worker_info = torch.utils.data.get_worker_info()
      if worker_info is None:
        self.worker_id = 0
        self.num_workers = 1
      else:
        self.worker_id = worker_info.id
        self.num_workers = worker_info.num_workers
      return dict(rank=self.rank,
              world_size=self.world_size,
              worker_id=self.worker_id,
              num_workers=self.num_workers)

  def set_epoch(self, epoch):
      self.epoch = epoch

  def sample(self, data):
      data = list(range(len(data)))
      if self.partition:
        if self.shuffle:
          random.Random(self.epoch).shuffle(data)
        data = data[self.rank::self.world_size]
      data = data[self.worker_id::self.num_workers]
      return data

class DataList(IterableDataset):
  def __init__(self, lists, shuffle=True, partition=True):
    self.lists = lists
    self.sampler = DistributedSampler(shuffle, partition)

  def set_epoch(self, epoch):
    self.sampler.set_epoch(epoch)

  def __iter__(self):
    sampler_info = self.sampler.update()
    indexes = self.sampler.sample(self.lists)
    for index in indexes:
      data = dict(src=self.lists[index])
      data.update(sampler_info)
      yield data

def IterDataset(data, feat_type, label, segment_dir):
  # segment_dir = '/tmp/LXKDATA/data/230617_segmen_data'
  DP = DataProcessor(label, segment_dir, 450)
  raw_dict_list = list(data.T.to_dict().values())
  dataset = DataList(raw_dict_list, False, False)
  if feat_type == 'stft':
    dataset = Processor(dataset, DP.compute_stft)
  elif feat_type == 'fbank':
    dataset = Processor(dataset, DP.compute_fbank)
  elif feat_type == 'mfcc':
    dataset = Processor(dataset, DP.compute_mfcc)
  elif feat_type == 'cqcc' or feat_type == 'ivector' or feat_type == 'vhubert' or feat_type == 'hubert' or 'loop' in feat_type:
    dataset = Processor(dataset, DP.get_computed_feat, feat_type)
  elif feat_type == 'melspecsegment':
    dataset = Processor(dataset, DP.compute_melspec, segment_dir)
  elif feat_type == 'stftsegment':
    dataset = Processor(dataset, DP.compute_segment_stft, segment_dir)
  elif feat_type == 'vhubertsegment':
    dataset = Processor(dataset, DP.get_segment_vhubert, segment_dir)
  elif feat_type == 'cmlrv':
    dataset = Processor(dataset, DP.get_cmlrv)
  elif feat_type == 'papi':
    dataset = Processor(dataset, DP.get_papi)
  elif feat_type == 'lip_cmlrv':
    dataset = Processor(dataset, DP.get_lipcmlrv)
  elif feat_type == 'lip':
    dataset = Processor(dataset, DP.get_lip)
  elif feat_type == 'papi_lip':
    dataset = Processor(dataset, DP.get_papilip)
  elif feat_type == 'papi_cropavi':
    dataset = Processor(dataset, DP.get_papicropavi)
  elif feat_type == 'papi_cmrlv':
    dataset = Processor(dataset, DP.get_papicmlrv)
  elif feat_type == 'papi_lipcmrlv':
    dataset = Processor(dataset, DP.get_papilipcmlrv)
  elif feat_type == 'phonation':
    dataset = Processor(dataset, DP.get_phonation)
  elif feat_type == 'articulation':
    dataset = Processor(dataset, DP.get_articulation)
  elif feat_type == 'prosody':
    dataset = Processor(dataset, DP.get_prosody)
  elif feat_type == 'cropavi':
    dataset = Processor(dataset, DP.get_cropavi)
  elif 'segment' in feat_type:
    dataset = Processor(dataset, DP.get_segment_feat, feat_type, segment_dir)
  return dataset

def AVDataset(modality, data, label, segment_dir="/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/data/segment_data"):
  DP = DataProcessor(modality, label, segment_dir, 450)
  raw_dict_list = list(data.T.to_dict().values())
  dataset = DataList(raw_dict_list, False, False)
  dataset = Processor(dataset, DP.read_loop_feat_avi)
  return dataset

def AVDataBatch(samples):
  audio_modality, video_modality = False, False
  if samples[0]['loop_feat'] != None:
    audio_modality = True
  if samples[0]['video_vowels'] != None:
    video_modality = True
  batch_loop_feat, batch_avi_data, batch_label = [], [], []
  if video_modality:
    max_len = max(tensor.shape[0] for sublist in samples for tensor in sublist['video_vowels'])
  for sample in samples:
    if audio_modality:
      batch_loop_feat.append(sample['loop_feat'])
    else:
      batch_loop_feat.append(torch.zeros(20))
    if video_modality:
      six_vowels_list = []
      for video_tensor in sample['video_vowels']:
        zeros_to_pad = torch.zeros((max_len-video_tensor.shape[0], *tuple(video_tensor.size()[1:])))
        padded_tensor =  torch.cat((video_tensor, zeros_to_pad), dim=0)
        trans_video_tensor = torch.transpose(padded_tensor, 0, 1)
        six_vowels_list.append(trans_video_tensor.unsqueeze(0))
      six_vowels_tensor = torch.cat(six_vowels_list, dim=0)
      batch_avi_data.append(six_vowels_tensor)
    else:
      batch_avi_data.append(torch.zeros(6, 1, 1, 80, 80))
    batch_label.append(sample['label'])
  batch_loop_feat_tensor = torch.stack(batch_loop_feat)
  batch_avi_data_tensor = torch.stack(batch_avi_data)
  batch_label_tensor = torch.stack(batch_label)
  return batch_loop_feat_tensor, batch_avi_data_tensor, batch_label_tensor


if __name__ == "__main__":
  import time
  start_time = time.time()
  csv_path = 'tmp/test_data.csv'
  raw_acu_feats = pd.read_csv(csv_path)

  base_dataset = IterDataset(raw_acu_feats, 'papi_cmrlv', 'Frenchay', 'data/segment_data/')
  data_loader = torch.utils.data.DataLoader(base_dataset, batch_size=8, num_workers=0)
  for i, batch in enumerate(data_loader):
    print(i, batch[0].shape, batch[1].shape)
    if i == 10:
      break
  
  end_time = time.time()
  print("duraion: {:.2f}sesonds".format(end_time - start_time))