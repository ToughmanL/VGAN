#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

class WriteGMMResult():
  def __init__(self) -> None:
    self.cons_gmm =None
    self.vowel_gmm = None
    self.feat_list = []
  
  def _load_gmm(self, cons_gmm_path, vowel_gmm_path):
    self.cons_gmm = joblib.load(cons_gmm_path)
    self.vowel_gmm = joblib.load(vowel_gmm_path)

  def _regulation(self, hyp_list):
    mid_idx = int(len(hyp_list)*2/3)
    last_cons_idx = 0
    while hyp_list[mid_idx] == 0:
      mid_idx -= 1
    for i in range(mid_idx, -1, -1):
      if hyp_list[i] == 0:
        last_cons_idx = i
        break
    for i in range(len(hyp_list)):
      if i <= last_cons_idx:
        hyp_list[i] = 0
      else:
        hyp_list[i] = 1
    return hyp_list, last_cons_idx

  def _model_inference(self, feat_syllable):
    hyp_list= []
    feats = feat_syllable.T
    score1 = self.cons_gmm.score_samples(feats)
    score2 = self.vowel_gmm.score_samples(feats)
    for k in range(len(score1)):
      if score1[k] > score2[k]:   # 为什么要大于score2[1]? 而不是score2[k]
        hyp_list.append(0)
      else:
        hyp_list.append(1)
    # 保存全为0的样本
    if (np.array(hyp_list) == 0).sum() == len(hyp_list):
      # GMM 无法判断的交给后续的pitch
      frame_result = 0
    else:
      # 找到元音开始时间
      _, vowel_index = self._regulation(hyp_list)
      frame_result = vowel_index * 0.01
    return frame_result

  def writeresult(self, cons_gmm_path, vowel_gmm_path, one_wav_feats):
    vowel_start_time = []
    self._load_gmm(cons_gmm_path, vowel_gmm_path)
    for feat_syllable in one_wav_feats:
      frame_result = self._model_inference(feat_syllable)
      vowel_start_time.append(frame_result)
    return vowel_start_time

if __name__ == '__main__':

  modeldir = "data/model/"
  cons_gmm_path = modeldir + "gmm_NS_conso_com100_max20_covfull.smn"
  vowel_gmm_path = modeldir + "gmm_NS_vowel_com80_max20_covfull.smn"
  result_path = "result/230617_gmm_con_vowel_segment.csv"
  multi_num = 60
  WG = WriteGMMResult()
  WG.writeresult(cons_gmm_path, vowel_gmm_path, result_path)


  