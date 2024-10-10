#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 data_processor.py
* @Time 	:	 2023/07/21
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import logging
import json
import random
import torch
import torchaudio
import cv2
import numpy as np

import scipy.io.wavfile as wavfile
from spafe.features.cqcc import cqcc

from torchvision.io import read_video
import torchvision.transforms.functional as F

class DataProcessor():
  def __init__(self, label, setment_dir, target_len) -> None:
    self.name_path_dict = {}
    self.label = label
    self.batch_size = 16
    self.target_len = target_len
    # self._get_name_path(setment_dir)
    self.setment_dir = setment_dir

  def _get_name_path(self, setment_dir):
    for root, dirs, files in os.walk(setment_dir, followlinks=True):
      for file in files:
        if len(file.split('.')) < 2:
          continue
        name, suff = file.split('.')[0], file.split('.')[1]
        if suff == 'wav':
          self.name_path_dict[name] = os.path.join(root, file)
  
  def _get_file_path(self, file_name, suffix):
    name_ll = file_name.split('_')
    if len(name_ll) == 7:
      pass
    elif len(name_ll) == 8:
      name_ll = name_ll[1:]
    else:
      print(file_name, 'error file')
      exit(-1) 
    N_S_name = "Control" if name_ll[0] == "N" else "Patient"
    person_name = name_ll[0] + '_' + name_ll[2] + '_' + name_ll[1]
    file_path = os.path.join(self.setment_dir, N_S_name, person_name, file_name + '.' + suffix)
    return file_path

  def _crop_lip(self, input_tensor, crop_size):
    h, w, d = input_tensor.shape
    crop_w, crop_d = crop_size

    start_w = (w - crop_w) // 2
    end_w = start_w + crop_w

    start_d = (d - crop_d) // 2
    end_d = start_d + crop_d

    cropped_tensor = input_tensor[:, start_w:end_w, start_d:end_d]
    return cropped_tensor

  def _read_and_convert_to_grayscale(self, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break
      # Convert to grayscale
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Convert to PyTorch tensor
      tensor_frame = F.to_tensor(gray_frame)
      # Normalize the tensor (optional)
      tensor_frame = F.normalize(tensor_frame, [0.5], [0.5])
      if tensor_frame.shape[2] > 80:
        tensor_frame = self._crop_lip(tensor_frame, (80, 80))
      frames.append(tensor_frame)
    cap.release()
    if len(frames) == 0:
      tensor_frame = torch.rand(1,80,80)
      frames.append(tensor_frame)
      frames.append(tensor_frame)
    return torch.stack(frames)

  def _sequence_pad(self, feat, target_len):
    if feat.shape[0] > target_len:
      feat = feat[:target_len,:]
    else:
      pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, target_len-feat.shape[0]))
      feat = pad(feat)
    return feat
  
  def video_pad(self, video_tensor, target_len):
    if video_tensor.shape[0] > target_len:
      padded_video_data = video_tensor[:target_len,:,:]
    else:
      padded_video_data = torch.nn.functional.pad(video_tensor, (0, 0, 0, 0, 0, target_len-video_tensor.shape[0]), 'constant', 0)
    return padded_video_data

  def compute_fbank(self, data,
              num_mel_bins=23,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_rate= 16000):
    # waveform = waveform * (1 << 15)
    random.seed(0)
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    for dataframe_dict in data:
      label_score = torch.as_tensor(dataframe_dict['src'][self.label])
      fbank_list = []
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict['src'][vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        # wav_path = self.name_path_dict[wav_name]
        wav_path = self._get_file_path(wav_name, 'wav')
        waveform, sr = torchaudio.load(wav_path)
        fbank = torchaudio.compliance.kaldi.fbank(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=0.0,
                    sample_frequency=sample_rate)
        if fbank.shape[1] > self.target_len:
          fbank = fbank[:,:self.target_len]
        else:
          pad = torch.nn.ZeroPad2d(padding=(0, self.target_len-fbank.shape[1], 0, 0))
          fbank = pad(fbank)
        fbank_list.append(fbank)
      fbank_feat = torch.cat(fbank_list, 1)
      yield fbank_feat, label_score

  def compute_mfcc(self, data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=13,
                 high_freq=0.0,
                 low_freq=20.0,
                 sample_rate=16000):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    target_len = 82
    for dataframe_dict in data:
      label_score = torch.as_tensor(dataframe_dict['src'][self.label])
      mfcc_list = []
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict['src'][vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        # wav_path = self.name_path_dict[wav_name]
        wav_path = self._get_file_path(wav_name, 'wav')
        waveform, sr = torchaudio.load(wav_path)
        # Only keep key, feat, label
        mfcc = torchaudio.compliance.kaldi.mfcc(waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    num_ceps=num_ceps,
                    high_freq=high_freq,
                    low_freq=low_freq,
                    sample_frequency=sample_rate)
        if mfcc.shape[0] > target_len:
          pad_mfcc = mfcc[:target_len,:]
        else:
          pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, target_len-mfcc.shape[0]))
          pad_mfcc = pad(mfcc)
        # first_diff_mfcc = torch.diff(pad_mfcc, n=1, dim=0)
        # second_diff_mfcc = torch.diff(first_diff_mfcc, n=1, dim=0)
        # concat_mfcc = torch.cat((pad_mfcc[:80,:], first_diff_mfcc[:80,:], second_diff_mfcc), dim=1)
        # mfcc_list.append(torch.transpose(concat_mfcc,0,1))
        mfcc_list.append(torch.transpose(pad_mfcc[:80,:],0,1))
      mfcc_feat = torch.stack(mfcc_list, 0)
      yield mfcc_feat, label_score.float()

  def compute_stft(self, data):
    loop = False
    if loop:
      random.seed(0)
      select_vowel_list = random.sample(['a', 'o', 'e', 'i', 'u', 'v'], 1)
      for dataframe_dict in data:
        label_score = torch.as_tensor(dataframe_dict['src'][self.label])
        select_feat_path = dataframe_dict['src'][select_vowel_list[0]+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        # wav_path = self.name_path_dict[wav_name]
        wav_path = self._get_file_path(wav_name, 'wav')
        waveform, sr = torchaudio.load(wav_path)
        stft_complex = torch.stft(waveform, 1139, hop_length=24, win_length=32, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        stft = stft_complex[:,:,:,0] # 取复数的实部
        target_len = 450
        if stft.shape[2] < target_len:
          pad = torch.nn.ZeroPad2d(padding=(0, target_len-stft.shape[2], 0, 0))
          stft = pad(stft)
        elif stft.shape[2] > target_len:
          stft = stft[:,:,:target_len]
        yield (stft.to(torch.float32), label_score.to(torch.float32))
    else:
      for dataframe_dict in data:
        obj = dataframe_dict['src']
        label_score = torch.as_tensor(obj[self.label])
        wav_path = obj['Path']
        waveform, sr = torchaudio.load(wav_path)
        stft_complex = torch.stft(waveform, 1139, hop_length=24, win_length=32, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        stft = stft_complex[:,:,:,0] # 取复数的实部
        target_len = 450 # 按照文章标准
        if stft.shape[2] < target_len:
          pad = torch.nn.ZeroPad2d(padding=(0, target_len-stft.shape[2], 0, 0))
          stft = pad(stft)
        elif stft.shape[2] > target_len:
          stft = stft[:,:,:target_len]
        yield (stft.to(torch.float32), label_score.to(torch.float32))

  def compute_segment_stft(self, data, feat_dir):
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      feat_path = os.path.join(feat_dir, dataframe_dict['Segname'] + '.wav')
      waveform, sr = torchaudio.load(feat_path, normalize=True)
      stft_complex = torch.stft(waveform, n_fft=255, hop_length=24, win_length=32, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
      stft = stft_complex[:,:,:,0] # 取复数的实部
      stft = stft.squeeze(0).transpose(0, 1)
      feat_data = self._sequence_pad(stft, target_len=128)
      if test_flag:
        return feat_data, label_score.float()
      else:
        yield feat_data, label_score.float()

  def compute_melspec(self, data, feat_dir):
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      feat_path = os.path.join(feat_dir, dataframe_dict['Segname'] + '.wav')
      waveform, sr = torchaudio.load(feat_path, normalize=True)
      melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, win_length=400, hop_length=160, n_mels=64)(waveform)
      melspec = melspec.squeeze(0).transpose(0, 1)
      feat_data = self._sequence_pad(melspec, target_len=64)
      if test_flag:
        return feat_data, label_score.float()
      else:
        yield feat_data, label_score.float()
  
  def get_computed_feat(self, data, feat_type):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    target_len = 60
    test_flag = False
    if feat_type == 'cqcc':
      suffix = 'cqcc.pt'
    elif feat_type == 'ivector':
      suffix = 'ivector.pt'
    elif 'wav2vec' in feat_type:
      suffix = 'w2v_max.pt'
    elif feat_type == 'hubert':
      suffix = 'hub_max.pt'
    elif feat_type == 'vhubert':
      suffix = 'vhubert.npy'
    else:
      print('feat type error')
      exit(-1)
    for dataframe_dict in data:
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True

      label_score = torch.as_tensor(dataframe_dict[self.label])
      feat_list = []
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict[vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        feat_path = self._get_file_path(wav_name, suffix)
        if '.pt' in feat_path:
          feat_data = torch.load(feat_path, map_location=torch.device('cpu')).float()
        elif '.npy' in feat_path: # vhubert
          feat_data = torch.from_numpy(np.load(feat_path)).float()
          # feat_data = torch.max(feat_data, dim=0).values
          feat_data = torch.mean(feat_data, dim=0)
        if feat_type == 'cqcc':
          feat_data = self._sequence_pad(feat_data, target_len)
        feat_list.append(feat_data)
      aoeiuv_feat_data = torch.stack(feat_list)
      if test_flag:
        return aoeiuv_feat_data, label_score.float()
      else:
        yield aoeiuv_feat_data, label_score.float()
  
  def get_segment_feat(self, data, feat_type, feat_dir):
    test_flag = False
    if feat_type == 'cqccsegment':
      suffix = 'cqcc.pt'
    elif feat_type == 'ivectorsegment':
      suffix = 'ivector.pt'
    elif feat_type == 'wav2vecsegment':
      suffix = 'w2v_max.pt'
    elif feat_type == 'mfccsegment':
      suffix = 'mfcc.pt'
    elif feat_type == 'hubertsegment':
      suffix = 'npy'
    else:
      print('feat type error')
      exit(-1)
    for dataframe_dict in data:
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True

      label_score = torch.as_tensor(dataframe_dict[self.label])
      feat_path = os.path.join(feat_dir, dataframe_dict['Segname'] + '.' + suffix)
      if feat_type == 'hubertsegment':
        feat_data = torch.from_numpy(np.load(feat_path)).float()
        feat_data = torch.mean(feat_data, dim=0)
      else:
        feat_data = torch.load(feat_path, map_location=torch.device('cpu')).float()

      if feat_type == 'cqccsegment' or feat_type == 'mfccsegment':
        feat_data = self._sequence_pad(feat_data, target_len=60)
      if test_flag:
        return feat_data, label_score.float()
      else:
        yield feat_data, label_score.float()

  def get_segment_vhubert(self, data, feat_dir):
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      feat_path = os.path.join(feat_dir, dataframe_dict['Segname'] + '.npy')
      feat_data = torch.from_numpy(np.load(feat_path)).float()
      feat_data = torch.mean(feat_data, dim=0)
      if test_flag:
        return feat_data, label_score.float()
      else:
        yield feat_data, label_score.float()

  def get_papi(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    common_feats = ['Jitter','Shimmer','HNR','gne','vfer','F1_sd','F2_sd','F3_sd','Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur','gop_con','gop_vow']
    arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      arti_feat_list = []
      for af in arti_feats:
        arti_feat_list.append(dataframe_dict[af])
      all_feat_list = []
      for vowel in select_vowel_list:
        # common feats
        comm_feat_list = []
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        arti_feat_tensor = torch.as_tensor(arti_feat_list)
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_data_tensor = torch.cat((comm_feat_tensor, arti_feat_tensor)) # 1, 20
        all_feat_list.append(all_data_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      if test_flag:
        return padded_sequence, label_score.float()
      else:
        yield padded_sequence, label_score.float()

  def get_cmlrv(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      cmlrv_list = []
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict[vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        cmlrv_path = self._get_file_path(wav_name, 'mean_cmlrv.pt')
        cmlrv_data = torch.load(cmlrv_path, map_location=torch.device('cpu'))
        cmlrv_data = (cmlrv_data - -0.0116) / 0.5150
        cmlrv_list.append(cmlrv_data)
      aoeiuv_cmlrv_data = torch.stack(cmlrv_list)
      if test_flag:
        return aoeiuv_cmlrv_data, label_score.float()
      else:
        yield aoeiuv_cmlrv_data, label_score.float()
    # mean: tensor(-0.0116) std: tensor(0.5150)
  
  def get_papicmlrv(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    common_feats = ['Jitter','Shimmer','HNR','gne','vfer','F1_sd','F2_sd','F3_sd','Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur','gop_con','gop_vow']
    arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']
    for dataframe_dict in data:
      dataframe_dict = dataframe_dict['src']
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      # arti feats
      arti_feat_list = []
      for af in arti_feats:
        arti_feat_list.append(dataframe_dict[af])
      for vowel in select_vowel_list:
        # cmlrv data
        select_feat_path = dataframe_dict[vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        cmlrv_path = self._get_file_path(wav_name, 'mean_cmlrv.pt')
        cmlrv_data = torch.load(cmlrv_path, map_location=torch.device('cpu'))
        cmlrv_data = (cmlrv_data - -0.0116) / 0.5150
        # common feats
        comm_feat_list = []
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        arti_feat_tensor = torch.as_tensor(arti_feat_list)
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_data_tensor = torch.cat((cmlrv_data, comm_feat_tensor, arti_feat_tensor))
        all_feat_list.append(all_data_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      yield padded_sequence, label_score.float()
  
  def get_lipcmlrv(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    lip_feats = ['alpha_stability','beta_stability','dist_stability','alpha_speed','beta_speed','inner_speed','inner_dist_min','inner_dist_max','w_min','w_max']
    for dataframe_dict in data:
      dataframe_dict = dataframe_dict['src']
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      for vowel in select_vowel_list:
        # cmlrv data
        select_feat_path = dataframe_dict[vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        cmlrv_path = self._get_file_path(wav_name, 'mean_cmlrv.pt')
        cmlrv_data = torch.load(cmlrv_path, map_location=torch.device('cpu'))
        cmlrv_data = (cmlrv_data - -0.0116) / 0.5150
        # common feats
        comm_feat_list = []
        for cf in lip_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_data_tensor = torch.cat((cmlrv_data, comm_feat_tensor))
        all_feat_list.append(all_data_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      yield padded_sequence, label_score.float()
  
  def get_lip(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    lip_feats = ['alpha_stability','beta_stability','dist_stability','alpha_speed','beta_speed','inner_speed','inner_dist_min','inner_dist_max','w_min','w_max']
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      for vowel in select_vowel_list:
        # common feats
        comm_feat_list = []
        for cf in lip_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_feat_list.append(comm_feat_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      if test_flag:
        return padded_sequence, label_score.float()
      else:
        yield padded_sequence, label_score.float()
  
  def get_papilip(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    lip_feats = ['alpha_stability','beta_stability','dist_stability','alpha_speed','beta_speed','inner_speed','inner_dist_min','inner_dist_max','w_min','w_max']
    common_feats = ['Jitter','Shimmer','HNR','gne','vfer','F1_sd','F2_sd','F3_sd','Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur','gop_con','gop_vow']
    arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      arti_feat_list = []
      for af in arti_feats:
        arti_feat_list.append(dataframe_dict[af])
      all_feat_list = []
      for vowel in select_vowel_list:
        # common feats
        lip_feat_list, comm_feat_list = [], []
        for cf in lip_feats:
          lip_feat_list.append(dataframe_dict[vowel+'-'+cf])
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        arti_feat_tensor = torch.as_tensor(arti_feat_list)
        lip_feat_tensor = torch.as_tensor(lip_feat_list)
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_data_tensor = torch.cat((comm_feat_tensor, arti_feat_tensor, lip_feat_tensor)) # 1, 30
        all_feat_list.append(all_data_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      if test_flag:
        return padded_sequence, label_score.float()
      else:
        yield padded_sequence, label_score.float()

  def get_papilipcmlrv(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    lip_feats = ['alpha_stability','beta_stability','dist_stability','alpha_speed','beta_speed','inner_speed','inner_dist_min','inner_dist_max','w_min','w_max']
    common_feats = ['Jitter','Shimmer','HNR','gne','vfer','F1_sd','F2_sd','F3_sd','Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur','gop_con','gop_vow']
    arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']
    for dataframe_dict in data:
      dataframe_dict = dataframe_dict['src']
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      arti_feat_list = []
      for af in arti_feats:
        arti_feat_list.append(dataframe_dict[af])
      for vowel in select_vowel_list:
        # cmlrv data
        select_feat_path = dataframe_dict[vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        cmlrv_path = self._get_file_path(wav_name, 'mean_cmlrv.pt')
        cmlrv_data = torch.load(cmlrv_path, map_location=torch.device('cpu'))
        cmlrv_data = (cmlrv_data - -0.0116) / 0.5150
        # lip feats
        lip_feat_list, comm_feat_list = [], []
        for cf in lip_feats:
          lip_feat_list.append(dataframe_dict[vowel+'-'+cf])
        # common feats
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        arti_feat_tensor = torch.as_tensor(arti_feat_list)
        lip_feat_tensor = torch.as_tensor(lip_feat_list)
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_data_tensor = torch.cat((cmlrv_data, lip_feat_tensor, comm_feat_tensor, arti_feat_tensor))
        all_feat_list.append(all_data_tensor)
      padded_sequence = torch.stack(all_feat_list)
      yield padded_sequence, label_score.float()

  def get_cropavi(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    for dataframe_dict in data:
      dataframe_dict = dataframe_dict['src']
      label_score = torch.as_tensor(dataframe_dict[self.label])
      cropavi_list = []
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict[vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        cropavi_path = self._get_file_path(wav_name, 'crop.pt')
        cropavi_data = torch.load(cropavi_path, map_location=torch.device('cpu'))
        cropavi_data = cropavi_data.squeeze(1)
        cropavi_data = self.video_pad(cropavi_data, 12) # 12 frames
        cropavi_list.append(cropavi_data.float())
      aoeiuv_cropavi_data = torch.stack(cropavi_list)
      yield aoeiuv_cropavi_data, label_score.float()

  def get_papicropavi(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    common_feats = ['Jitter','Shimmer','HNR','gne','vfer','F1_sd','F2_sd','F3_sd','Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur','gop_con','gop_vow']
    arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list, arti_feat_list = [], []
      for af in arti_feats:
        arti_feat_list.append(dataframe_dict[af])
      for vowel in select_vowel_list:
        select_feat_path = dataframe_dict[vowel+ '_mfcc']
        wav_name = select_feat_path.split('.')[0]
        # cropavi data
        cropavi_path = self._get_file_path(wav_name, 'crop.pt')
        cropavi_data = torch.load(cropavi_path, map_location=torch.device('cpu'))
        cropavi_data = cropavi_data.squeeze(1)
        cropavi_data = self.video_pad(cropavi_data, 12) # 12 frames
        # papi data
        comm_feat_list = []
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        arti_feat_tensor = torch.as_tensor(arti_feat_list)
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        papi_data = torch.cat((comm_feat_tensor, arti_feat_tensor))
        padded_tensor = torch.zeros(cropavi_data.shape[-2], cropavi_data.shape[-1])
        padded_tensor[:papi_data.shape[0], :papi_data.shape[0]] = papi_data.view(-1, 1)
        all_data_tensor = torch.cat((cropavi_data.float(), padded_tensor.unsqueeze(0)))
        all_feat_list.append(all_data_tensor)
      aoeiuv_cropavi_data = torch.stack(all_feat_list)
      if test_flag:
        return aoeiuv_cropavi_data, label_score.float()
      else:
        yield aoeiuv_cropavi_data, label_score.float()

  def get_phonation(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    common_feats = ['Jitter','Shimmer','HNR','gne','vfer']
    for dataframe_dict in data:
      test_flag = False
      if 'src' in dataframe_dict:
        dataframe_dict = dataframe_dict['src']
      else:
        test_flag = True
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      for vowel in select_vowel_list:
        # common feats
        comm_feat_list = []
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_feat_list.append(comm_feat_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      if test_flag:
        return padded_sequence, label_score.float()
      else:
        yield padded_sequence, label_score.float()
  
  def get_articulation(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    common_feats = ['F1_sd','F2_sd','F3_sd', 'Intensity_sd']
    arti_feats = ['tougue_dist','jaw_dist','move_degree','VSA','VAI','FCR']
    for dataframe_dict in data:
      dataframe_dict = dataframe_dict['src']
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      # arti feats
      arti_feat_list = []
      for af in arti_feats:
        arti_feat_list.append(dataframe_dict[af])
      for vowel in select_vowel_list:
        # common feats
        comm_feat_list = []
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        arti_feat_tensor = torch.as_tensor(arti_feat_list)
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_data_tensor = torch.cat((comm_feat_tensor, arti_feat_tensor))
        all_feat_list.append(all_data_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      yield padded_sequence, label_score.float()
  
  def get_prosody(self, data):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    common_feats = ['Intensity_mean','Intensity_sd','Vowel_dur','Syllable_dur']
    for dataframe_dict in data:
      dataframe_dict = dataframe_dict['src']
      label_score = torch.as_tensor(dataframe_dict[self.label])
      all_feat_list = []
      for vowel in select_vowel_list:
        # common feats
        comm_feat_list = []
        for cf in common_feats:
          comm_feat_list.append(dataframe_dict[vowel+'-'+cf])
        comm_feat_tensor = torch.as_tensor(comm_feat_list)
        all_feat_list.append(comm_feat_tensor)
      padded_sequence = torch.stack(all_feat_list)
      # print(padded_sequence.shape)
      yield padded_sequence, label_score.float()

  def read_loop_feat_avi(self, data, modality):
    select_vowel_list = ['a', 'o', 'e', 'i', 'u', 'v']
    arti_feats = ['tougue_dist', 'jaw_dist', 'move_degree', 'VSA', 'VAI', 'FCR']
    comm_feats = ['Jitter', 'Shimmer', 'HNR', 'gne', 'vfer', 'F1_sd', 'F2_sd', 'F3_sd', 'Intensity_mean', 'Intensity_sd', 'Vowel_dur', 'Syllable_dur', 'gop_con','gop_vow']
    for dataframe_dict in data:
      feat_dict = dataframe_dict['src']
      # feat_dict = dataframe_dict
      video_vowel_list, loop_vowel_list = [], []
      for vowel in select_vowel_list:
        # get label
        label_score = torch.as_tensor(feat_dict[self.label])
        if 'video' in modality:
          # get avi data
          avi_name = feat_dict[vowel + '_mfcc'].split('.')[0]
          # avi_path = self.name_path_dict[avi_name].replace('.wav', '.crop.avi')
          avi_path = self._get_file_path(avi_name, '.crop.avi')
          # video_tensor = self._read_and_convert_to_grayscale(avi_path)
          pt_path = self._get_file_path(avi_name, '.crop.pt')
          if os.path.exists(pt_path):
            video_tensor = torch.load(pt_path)
          else:
            video_tensor = torch.zeros(1,1,80,80)
          video_vowel_list.append(video_tensor.to(torch.float32))
        if 'audio' in modality:
          # get audio GRAPH data
          vowel_comfeat_name = [vowel + '-' + featname for featname in comm_feats]
          vowel_comfeat_value = [feat_dict[key] for key in vowel_comfeat_name if key in feat_dict]
          vowel_artfeat_value = [feat_dict[key] for key in arti_feats if key in feat_dict]
          vowel_comfeat_value.extend(vowel_artfeat_value)
          loop_vowel_list.append(torch.as_tensor(vowel_comfeat_value))
      loop_feat_tensor = None
      if 'audio' in modality:
        loop_feat_tensor = torch.stack(loop_vowel_list).to(torch.float32)
      if len(video_vowel_list) == 0:
        video_vowel_list = None

      yield {'loop_feat':loop_feat_tensor, 'video_vowels':video_vowel_list, 'label':label_score.to(torch.float32)}

  def padding(self, data):
    for stft, label_score in data:
      if stft.shape[2] < self.target_len:
        pad = torch.nn.ZeroPad2d(padding=(0, self.target_len-stft.shape[2], 0, 0))
        stft = pad(stft)
      elif stft.shape[2] > self.target_len:
        stft = stft[:,:,:self.target_len]
      yield stft, label_score

  def batch(self, data):
    feat_buf, label_buf = [], []
    for stft, label_score in data:
      feat_buf.append(stft)
      label_buf.append(label_score)
      if len(feat_buf) >= self.batch_size:
        yield torch.stack(feat_buf), torch.stack(label_buf)
        feat_buf, label_buf = [], []
    if len(feat_buf) > 0:
      yield torch.stack(feat_buf), torch.stack(label_buf)

def test_grey():
  def _crop_lip(input_tensor, crop_size):
    h, w, d = input_tensor.shape
    crop_w, crop_d = crop_size

    start_w = (w - crop_w) // 2
    end_w = start_w + crop_w

    start_d = (d - crop_d) // 2
    end_d = start_d + crop_d

    cropped_tensor = input_tensor[:, start_w:end_w, start_d:end_d]
    return cropped_tensor

  video_path = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/230617_segmen_data/Patient/S_00015_F/S_F_00015_G5_task8_1_9.crop.avi'
  cap = cv2.VideoCapture(video_path)
  frames = []
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert to PyTorch tensor
    tensor_frame = F.to_tensor(gray_frame)
    # Normalize the tensor (optional)
    tensor_frame = F.normalize(tensor_frame, [0.5], [0.5])
    if tensor_frame.shape[2] > 80:
      tensor_frame = _crop_lip(tensor_frame, (80, 80))
    if tensor_frame.shape[0] == 0:
      tensor_frame = torch.rand(1,80,80)
    frames.append(tensor_frame)
  cap.release()

def test_stft():
  wav_path = 'data/segment_data/Control/N_10001_F/N_F_10001_G1_task1_1_1.wav'
  waveform, sr = torchaudio.load(wav_path)
  Spectr = torchaudio.transforms.Spectrogram(n_fft=1139, hop_length=24, win_length=32, window_fn=torch.hann_window, power=None, center=True, pad_mode='reflect', normalized=False, onesided=True)(waveform)
  stft_complex = torch.stft(waveform, 1139, hop_length=24, win_length=32, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
  stft = stft_complex[:,:,:,0].to(torch.float32)

  print(stft.shape)
  target_len = 450
  if stft.shape[2] < target_len:
    pad = torch.nn.ZeroPad2d(padding=(0, target_len-stft.shape[2], 0, 0))
    stft = pad(stft)
  elif stft.shape[2] > target_len:
    stft = stft[:,:,:target_len]
  print(stft.shape)

def test_mfcc():
  wav_path = 'data/segment_data/Control/N_10001_F/N_F_10001_G1_task1_1_1.wav'
  waveform, sr = torchaudio.load(wav_path)
  target_len = 82
  mfcc = torchaudio.compliance.kaldi.mfcc(waveform,
                    num_mel_bins=23,
                    frame_length=25,
                    frame_shift=10,
                    dither=0.0,
                    num_ceps=13,
                    high_freq=0.0,
                    low_freq=20.0,
                    sample_frequency=16000)
  if mfcc.shape[0] > target_len:
    pad_mfcc = mfcc[:target_len,:]
  else:
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, target_len-mfcc.shape[0]))
    pad_mfcc = pad(mfcc)
  first_diff_mfcc = torch.diff(pad_mfcc, n=1, dim=0)
  second_diff_mfcc = torch.diff(first_diff_mfcc, n=1, dim=0)
  concat_mfcc = torch.cat((pad_mfcc[:80,:], first_diff_mfcc[:80,:], second_diff_mfcc), dim=1)
  print(torch.transpose(concat_mfcc,0,1).shape)

def test_loop_feat_avi():
  import pandas as pd
  csv_path = '/tmp/LXKDATA/result_intermediate/acoustic_loop_feats_0714.csv'
  raw_acu_feats = pd.read_csv(csv_path)
  row_dict_list = raw_acu_feats.T.to_dict().values()
  setment_dir="/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/data/segment_data"
  DP = DataProcessor("loop_feat_avi", "Frenchay", setment_dir, 50)
  DP.read_loop_feat_avi(row_dict_list)

def test_cmlrv():
  import pandas as pd
  csv_path = 'tmp/test_data.csv'
  raw_acu_feats = pd.read_csv(csv_path)
  row_dict_list = raw_acu_feats.T.to_dict().values()
  setment_dir="/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/data/segment_data"
  DP = DataProcessor("Frenchay",setment_dir, 50)
  # DP.get_cmlrv(row_dict_list)
  # DP.get_papicmlrv(row_dict_list)
  DP.get_cqcc(row_dict_list)

if __name__ == "__main__":
  # test_stft()
  # test_mfcc()
  # test_loop_feat_avi()
  # test_grey()
  test_cmlrv()