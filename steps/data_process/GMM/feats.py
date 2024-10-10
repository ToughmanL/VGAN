#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from utils.multi_process import MultiProcess

class FeatExtrct():
  def __init__(self, data, sr):
    self.frame_length = 0.04 # 40ms
    self.frame_shift = 0.01 # 30ms
    self.win_length = int(self.frame_length*sr)
    self.hop_length = int(self.frame_shift*sr)
    self.n_fft = 256
    self.data = data
    self.sr = sr

  def _pitch_extract(self):
    pitches, magnitudes = librosa.core.piptrack(y=self.data, sr=self.sr, \
      win_length=self.win_length, hop_length=self.hop_length, \
      threshold=0.1, fmin=75, fmax=1600)
    max_indexes = np.argmax(magnitudes, axis=0)
    pitches = pitches[max_indexes, range(magnitudes.shape[1])]
    return pitches

  def _mfcc_extract(self):
    mfcc = librosa.feature.mfcc(y=self.data, sr=self.sr, \
           win_length=self.win_length, hop_length=self.hop_length, \
           center=True, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc, mode="nearest")
    mfcc_delta2 = librosa.feature.delta(mfcc, mode="nearest", order=2)
    mfcc_delta12 = np.vstack((mfcc, mfcc_delta, mfcc_delta2))
    return mfcc_delta12

  def _zero_crossing_rate(self):
    zcr = librosa.feature.zero_crossing_rate(self.data, \
              frame_length=self.win_length, hop_length=self.hop_length)
    return zcr

  def get_feats(self):
    pitch = self._pitch_extract()
    mfcc = self._mfcc_extract()
    zcr = self._zero_crossing_rate()
    return np.vstack((pitch, mfcc, zcr))


class DataPrep():
  def __init__(self, csvordir, feat_dir):
    self.csvordir = csvordir
    self.wavlist = []
    self.feat_dir = feat_dir

  def _get_csv_wavs(self):
    df = pd.read_csv(self.csvordir, header=0, sep=",")
    self.wavlist = df.loc[:,["文件名", "序号", "音节起始时刻点", "韵母起始时刻点", "音节结束时刻点"]]

  def _get_dir_wavs(self):
      for root, dirs, files in os.walk(self.csvordir):
        for file in files:
          if '.wav' in file:
            self.wavlist.append(os.path.join(root, file))
  
  def _get_label(self, labelfile):
    with open(labelfile, 'r') as fp:
      return []
  
  def _get_feats(self, data, sr):
    FE = FeatExtrct(data, sr)
    feats = FE.get_feats()
    return feats

  def _feat_with_label(self, wav_path_index):
    i = wav_path_index
    # for i in range(0,1):
    conso_offset = self.wavlist.loc[i, '音节起始时刻点']
    conso_duration = self.wavlist.loc[i, '韵母起始时刻点'] - self.wavlist.loc[i, '音节起始时刻点']
    vowel_offset = self.wavlist.loc[i, '韵母起始时刻点']
    vowel_duration = self.wavlist.loc[i, '音节结束时刻点'] - self.wavlist.loc[i, '韵母起始时刻点']
    wav_path = self.wavlist.loc[i, "文件名"]
    person_name = wav_path.split("/")[-1].split(".")[0] + "_" + str(self.wavlist.loc[i, "序号"])
    # print("person_name : ", person_name)

    conso_data, _ = librosa.load(wav_path, sr=16000, offset=conso_offset, duration=conso_duration)
    vowel_data, _ = librosa.load(wav_path, sr=16000, offset=vowel_offset, duration=vowel_duration)
    final_feats = []
    # 辅音标签为0， 元音标签为1
    if conso_duration >= 0.05:
      conso_feats = self._get_feats(conso_data, 16000)
      conso_feats = np.r_[conso_feats, np.expand_dims(np.zeros(conso_feats[0].size),axis=0)]
      final_feats = conso_feats
    if vowel_duration >= 0.05:
      vowel_feats = self._get_feats(vowel_data, 16000)
      vowel_feats = np.r_[vowel_feats, np.expand_dims(np.ones(vowel_feats[0].size),axis=0)]
      if len(final_feats) == 0:
        final_feats = vowel_feats
      else:
        final_feats = np.hstack((final_feats, vowel_feats))
    # person_feats = np.hstack((conso_feats, vowel_feats))
    feat_path = self.feat_dir + "_" + person_name + ".npy"
    np.save(file=feat_path, arr=final_feats)
  
  def _feat_without_label(self, wav_path):
    wav_data, _ = librosa.load(wav_path, sr=16000)
    feats = self._get_feats(wav_data, 16000)
    wav_name = os.path.basename(wav_path)
    person_name = wav_path.split("/")[-2]
    if not os.path.exists(os.path.join(self.feat_dir, person_name)):
      os.makedirs(os.path.join(self.feat_dir, person_name))
    feat_path = os.path.join(self.feat_dir, person_name, wav_name)
    feat_path = feat_path.replace('.wav', '.npy')
    np.save(file=feat_path, arr=feats)
    # print(feat_path)
  
  def multi_process(self, withflag, multi_num):
    if withflag: # train and test
      self._get_csv_wavs()
      if multi_num == 1: # 线程数为1
        for wav_path_index in range(len(self.wavlist)):
          self._feat_with_label(wav_path_index)
      else: # 多线程
        MP = MultiProcess()
        MP.multi_not_result(func=self._feat_with_label, arg_list=range(len(self.wavlist)), process_num=multi_num)
    else: # implement
      self._get_dir_wavs()
      if multi_num == 1: # 线程数为1
        for wav_path in self.wavlist:
          self._feat_without_label(wav_path)
      else:
        MP = MultiProcess()
        MP.multi_not_result(func=self._feat_without_label, arg_list=self.wavlist, process_num=multi_num)

if __name__ == '__main__':
  withflag = False # True是训练，False是推理
  multi_num = 60
  if withflag:
    wavcsv = "data/textgrid/select_TextGrid.csv" # for train and test
    train_test_feat_path = "data/textgrid/train_test_feats"
    DP = DataPrep(wavcsv, train_test_feat_path)
    DP.multi_process(withflag, multi_num)
  else:
    wavdir = "/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/data/segment_data/Control" # for implement
    implement_feat_path = "/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/GMM/data/gmmfeat/Control"
    DP = DataPrep(wavdir, implement_feat_path)
    DP.multi_process(withflag, multi_num)