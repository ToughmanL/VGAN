#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 extract_eGeMAPs.py
* @Time 	:	 2023/03/14
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 compute egemaps
'''

import os
import csv
import numpy as np
from utils.multi_process import MultiProcess

opensmile_path = '/mnt/shareEEx/liuxiaokang/tool/opensmile-3.0-linux-x64/'
egemaps_conf = opensmile_path + 'config/egemaps/v01b/eGeMAPSv01b.conf'
# ComParE_conf = opensmile_path + 'config/compare16/ComParE_2016.conf'
pathExcuteFile = opensmile_path + 'bin/SMILExtract'

class ComputeEgemaps():
  def __init__(self, data_dir):
    self.wav_path_list = []
    self._get_wavs(data_dir)

  def _get_wavs(self, data_dir):
    for root, dirs, files in os.walk(data_dir):
      for file in files:
        if file[-4:] == '.wav':
          wav_path = os.path.join(root, file)
          self.wav_path_list.append(wav_path)
  
  def _egemaps_command(self, wav_path):
    egemaps_path = wav_path.replace('.wav', '.egemaps.csv')
    cmd = '{exec} -C {config} -I {wav} -csvoutput {egemaps}'.format(exec=pathExcuteFile, config=egemaps_conf, wav=wav_path, egemaps=egemaps_path)
    print(cmd)
    os.system(cmd)

  def get_egmaps(multinum):
    if multinum == 1:
      for wav_path in self.wav_path_list:
        self._egemaps_command(wav_path)
    else:
      MP = MultiProcess()

if __name__ == "__main__":
  segment_dir = '../data/segment_data/N_S_data'
  get_egmaps(segment_dir)