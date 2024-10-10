#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 phase_segment.py
* @Time 	:	 2024/02/21
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import pandas as pd
import numpy as np
import os, csv, sys
import torch
# import torchaudio
import torch.nn.functional as F
import soundfile as sf
from utils.get_files_dirs import FileDir
from utils.multi_process import MultiProcess
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',30)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfSupervision():
  def __init__(self, file_path, result_path, feat_type, device='cuda:0') -> None:
    self.model_flag = ''
    self.mfcc = None
    self.model = None
    self.cfg = None
    self.FD = FileDir()
    self.file_set = set()
    self.file_path = file_path
    self.result_path = result_path
    self.wav_info_list = self.get_wav_info(file_path)
    self.device = torch.device(device)
    self.feat_type = feat_type
    
  def get_wav_info(self, file_path):
    wav_info_list = []
    if os.path.isdir(file_path):
      for root, dirs, files in os.walk(file_path, followlinks=True):
        for file in files:
          file_suff = file.split('.', 1)[-1]
          if file_suff == 'wav':
            wav_info = {'Path':os.path.join(root, file), 'Start':0, 'End':0}
            wav_info_list.append(wav_info)
    elif os.path.isfile(file_path):
      df = pd.read_csv(file_path)
      list_of_dict = df.to_dict('records')
      for row in list_of_dict:
        tmp_dict = {}
        wav_name = os.path.basename(row['Path']).split('.')[0]
        time =  str(round(float(row['Start']),3)) + '_' + str(round(float(row['End']),3))
        tmp_dict['Segname'] = wav_name + '_' + time
        tmp_dict.update(row)
        wav_info_list.append(tmp_dict)
    else:
      exit(-1)
    return wav_info_list

  def _get_input(self, wav_info):
    wav_path = wav_info['Path']
    wav, sr = sf.read(wav_path)
    if wav_info['Start'] == wav_info['End']:
      start_frame = 0
      end_frame = -1
    else:
      start_frame = int(float(wav_info['Start']) * sr)
      end_frame = int(float(wav_info['End']) * sr)
    feat = torch.from_numpy(wav[start_frame:end_frame]).float()
    return inputs
  
  def interface_hidden_feat(self, model_path, multi_num=1):
    self._load_model(model_path)
    if multi_num != 1:
      MP = MultiProcess()
      MP.multi_not_result(func=self._extract_output, arg_list=self.wav_info_list, process_num=multi_num)
    else:
      for wav_path in self.wav_info_list:
        self._extract_output(wav_path)
    
    if os.path.isfile(self.file_path): # segment级别csv需要保存
      df = pd.DataFrame(self.wav_info_list)
      new_order = ['Person', 'Path', 'Segname', 'Start', 'End', 'Frenchay', 'fanshe', 'huxi', 'chun', 'he', 'ruane', 'hou', 'she', 'yanyu']
      df = df[new_order]
      df.to_csv(self.file_path.split('.')[0] + '_phasesegment.csv',index=False)
      print(len(self.file_set))
    

if __name__ == "__main__":
  wav_dir = sys.argv[1]
  device = sys.argv[2]
  feat_type = 'hubert'
  # wav_dir = "data/segment_data"
  # wav_dir = "data/result_intermediate/phase_setment_1119.csv"
  if feat_type == 'wav2vec':
    model_path = "pre_train_model/wav2vec2-base/chinese-wav2vec2-base.pt"
  elif feat_type == 'hubert':
    model_path = "pre_train_model/hubert-base/chinese-hubert-base.pt"
  
  result_path = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/dual-mode_feat/pretrain_feat/data/segment_data/PhaseSegData/'
  SP = SelfSupervision(wav_dir, result_path, feat_type, device)
  SP.interface_hidden_feat(model_path, 1)
  print("finished")

