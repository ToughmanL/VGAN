#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 baseline_feats.py
* @Time 	:	 2023/05/01
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 Prepare the training sample
'''

import os
import csv
import numpy as np
import pandas as pd
import librosa

from utils.get_files_dirs import FileDir
from utils.multi_process import MultiProcess

opensmile_path = '/mnt/shareEEx/liuxiaokang/tool/opensmile-3.0-linux-x64/'
egemaps_conf = opensmile_path + 'config/egemaps/v01b/eGeMAPSv01b.conf'
pathExcuteFile = opensmile_path + 'bin/SMILExtract'

class BaselineFeats:
  def __init__(self):
    self.wavpath_list = []
    self.frame_length = 0.025 # 25ms
    self.frame_shift = 0.01 # 10ms
    self.sr = 16000
    self.win_length = int(self.frame_length*self.sr)
    self.hop_length = int(self.frame_shift*self.sr)
  
  def _get_wavpath(self, datadir):
    for root, dirs, files in os.walk(datadir):
      for file in files:
        suff = file.split('.')[-1]
        if suff == 'wav':
          self.wavpath_list.append(os.path.join(root, file))
  
  def _compute_egemaps(self, wav_path):
    egemaps_path = wav_path.replace('.wav', '.egemaps.csv')
    cmd = '{exec} -C {config} -I {wav} -csvoutput {egemaps}'.format(exec=pathExcuteFile, config=egemaps_conf, wav=wav_path, egemaps=egemaps_path)
    os.system(cmd)
  
  def _compute_mfcc(self, wav_path):
    mfcc_path = wav_path.replace('.wav', '.mfcc.txt')
    data, _ = librosa.load(wav_path, sr=self.sr)
    mfcc = librosa.feature.mfcc(y=data, sr=self.sr, \
           win_length=self.win_length, hop_length=self.hop_length, \
           center=True, n_mfcc=13)
    mfcc_delta1 = librosa.feature.delta(mfcc, mode="nearest")
    mfcc_delta2 = librosa.feature.delta(mfcc, mode="nearest", order=2)
    mfcc_39 = np.vstack((mfcc, mfcc_delta1, mfcc_delta2))
    np.savetxt(mfcc_path, mfcc_39, delimiter=',')
  
  def _compute_stft(self, wav_path):
    stft_path = wav_path.replace('.wav', '.stft.txt')
    data, _ = librosa.load(wav_path, sr=self.sr)
    # 1139->570, 32->2ms, 24->1.5ms
    stft = librosa.stft(data, n_fft=1139, hop_length=24, win_length=32, window='hann', center=True, pad_mode='reflect', dtype='float32')
    np.savetxt(stft_path, stft, delimiter=',')

  def get_feats(self, datadir, feat_type, process_num=1):
    self._get_wavpath(datadir)
    if process_num == 1:
      for wav_path in self.wavpath_list:
        if feat_type == 'mfcc':
          self._compute_mfcc(wav_path)
        elif feat_type == 'egemaps':
          self._compute_egemaps(wav_path)
        elif feat_type == 'stft':
          self._compute_stft(wav_path)
    else:
      MP = MultiProcess()
      if feat_type == 'mfcc':
        MP.multi_not_result(self._compute_mfcc, self.wavpath_list, process_num)
      elif feat_type == 'egemaps':
        MP.multi_not_result(self._compute_egemaps, self.wavpath_list, process_num)
      elif feat_type == 'stft':
        MP.multi_not_result(self._compute_stft, self.wavpath_list, process_num)

class BasefeatSaving():
  def __init__(self):
    self.MP = MultiProcess()
    self.name_data_dict = {}
    self.person_label = {}
  
  def _read_csv(self, csv_path):
    data = pd.read_csv(csv_path).values
    return {'name':os.path.basename(csv_path), 'data':data}

  def _get_feats_file(self, featdir, feattype):
    FD = FileDir()
    FD.get_spec_files(featdir, feattype)
    featpath_list = FD.file_path_list
    data_list = self.MP.multi_with_result(func=self._read_csv, arg_list=featpath_list, process_num=30)
    for name_data in data_list:
      self.name_data_dict[name_data['name']] = name_data['data']
  
  def _task_enamble(self, vowel_file_dict):
    vowel_data_list = []
    for vowel_data in vowel_file_dict:
      name = vowel_data['name']
      vowel_data = self.name_data_dict[name]
      vowel_data_list.append(vowel_data)
    new_data = np.concatenate(vowel_data_list)
    label = self.person_label['name']
    return {'person':person, 'data':new_data, 'label':label}

  def _read_basefeat(self, base_feat_path, label_csv):
    basefeats = pd.read_csv(base_feat_path)
    self._read_label(label_csv) # 读取标签
    if self.com_norm:
      persons = basefeats['Person'].unique()
    print(len(persons))
    all_data_list = []
    for p in persons:
      person_data = basefeats[basefeats['Person'] == p].reset_index(drop=True)
      para_list = get_loop_data(person_data) # loop扩充数据
      data_list = self.MP.multi_with_result(func=self._task_enamble, arg_list=para_list, process_num=30)
      all_data_list = all_data_list + data_list
    with open(result_path, 'w') as fp:
      pickle.dump(all_data_list, f)


if __name__ == '__main__':
  datadir = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/230617_segmen_data/Patient/'
  test_wav = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/230617_segmen_data/Control/N_10001_F/N_F_10001_G1_task1_1_1.wav'
  BF = BaselineFeats()
  # BF.get_feats(datadir, 'mfcc', process_num=60)
  # BF.get_feats(datadir, 'egemaps', process_num=60)
  BF.get_feats(datadir, 'stft', process_num=60)
  # BF._compute_stft(test_wav)