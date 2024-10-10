#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import librosa
import librosa.display
import warnings
warnings.filterwarnings("ignore")

class FeatExtrct():
  def __init__(self, data, sr):
    self.frame_length = 0.04 # 40ms
    self.frame_shift = 0.01 # 30ms
    self.win_length = int(self.frame_length*sr)
    self.hop_length = int(self.frame_shift*sr)
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
    # (119,)
    # (39, 119)
    # (1, 119)
    pitch = self._pitch_extract()
    mfcc = self._mfcc_extract()
    zcr = self._zero_crossing_rate()
    return np.vstack((pitch, mfcc, zcr))


class DataPrep():
  def __init__(self):
    self.sr = 16000

  def _get_feats(self, data, sr):
    FE = FeatExtrct(data, sr)
    feats = FE.get_feats()
    return feats

  #提取一个音频文件的数据
  def feats_process(self, wav_info):
    wav_path = wav_info[0][0]
    wav_data, _ = librosa.load(wav_path, sr = self.sr)#对每一个采样点返回一个值
    #循环每个syllable
    one_wav_feats = []
    for syllable_wav in wav_info:
      wave_start = int(syllable_wav[1] * self.sr)
      wave_end = int(syllable_wav[2] * self.sr)
      syllable_wav_data = wav_data[wave_start : wave_end]
      feats = self._get_feats(syllable_wav_data, self.sr)
      one_wav_feats.append(feats)
    return one_wav_feats
  
if __name__ == '__main__':
  wav_info = []
  DP = DataPrep()
  DP.feats_process(wav_info)