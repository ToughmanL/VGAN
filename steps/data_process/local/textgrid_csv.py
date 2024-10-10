#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   textgrid_csv.py
@Time    :   2022/10/31 23:53:11
@Author  :   lxk 
@Version :   1.0
@Contact :   xk.liu@siat.ac.cn
@License :   (C)Copyright 2022-2025, lxk&AISML
@Desc    :   None
'''

import pandas as pd
import os
import wave
import cv2
import numpy as np
import utils.textgrid as tg
import utils.get_files_dirs as gfd
from utils.multi_process import MultiProcess
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(filename="log/textgrid_csv.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

class Text2Csv():
  """
  description  : convert textgrids in dir to csv 
  Args  : text_dir
  Return  : csv_path
  """
  def __init__(self, text_dir):
    self.text_files = []
    self.Syllable_Vowel = {}
    self.Syllable_Phone = {}
    self.Total_data = pd.DataFrame(columns=['Filename', 'Task', 'Session', 'Person','Path', 'Start', 'End', 'Text', 'Duration', 'Syllable', 'Vowel', 'count'])
    self._read_word2phone_dict()
    self._get_all_file(text_dir)
  
  def _read_word2phone_dict(self):
    # with open('./conf/Syllable2Vowel.txt', 'r', encoding='utf-8') as f:
    #   for line in f.readlines():
    #     line = line.strip('\n')
    #     syllable, vowel = line.split(' ')
    #     self.Syllable_Vowel[syllable] = vowel

    # with open('./conf/Syllable2Phone.txt', 'r', encoding='utf-8') as f:
    #   for line in f.readlines():
    #     line = line.strip('\n')
    #     syllable, phone = line.split(' ')
    #     self.Syllable_Phone[syllable] = phone
    vowels = ['a', 'o', 'e', 'i', 'u', 'v']
    with open('./conf/txt2phone.txt', 'r', encoding='utf-8') as fp:
      for line in fp:
        line = line.strip()
        ll = line.split(' ')
        vowel = ''
        for ch in ll[1:]:
          for vo in vowels:
            if vo in ch:
              vowel = ch
              break
          else:
            continue
          break
        self.Syllable_Vowel[ll[0]] = ''.join([i for i in vowel if not i.isdigit()])
        self.Syllable_Phone[ll[0]] = ''.join(ll[1:])

  def _get_hanzi(self, text):
    new_text = ''
    for ch in text:
      if '\u4e00' <= ch <= '\u9fa5':
        new_text += ch
    return new_text
  
  def _get_all_file(self, text_dir):
    for root, dirs, files in os.walk(text_dir, followlinks=True):
      for file in files:
        if 'task5_1' in file:
          continue
        file_suff = file.split('.', 1)[-1]
        if file_suff == 'TextGrid':
          self.text_files.append(os.path.join(root, file))

  def _avi_frame_check(self, wav_path):
    avi_path = wav_path.replace('.wav', '.avi')

    video = cv2.VideoCapture(avi_path)
    video_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)

    f = wave.open(wav_path)
    audio_time = f.getparams().nframes / f.getparams().framerate
    theoretical_frame = int(audio_time * 30) + 1
    diff_frame = theoretical_frame - video_frame
    return diff_frame

  def lsj_text_csv(self): # convert textgrid to csv
    for text_path in self.text_files: # Process one by one
      txt_tier = tg.read_textgrid(text_path, 'TEXT')
      error_tier = tg.read_textgrid(text_path, 'ERROR')
      filename = os.path.basename(text_path)
      tmp = filename.split('.')[0].split('_')
      task = tmp[-3] + '_' + tmp[-2] + '_' + tmp[-1]
      person = tmp[0] + '_' + tmp[2] + '_' + tmp[1] if tmp[0] != 'repeat' else tmp[1] + '_' + tmp[3] + '_' + tmp[2]
      session = person + '_' + tmp[-3]

      if len(txt_tier) != len(error_tier):
        logging.error('Please check the length of error_tier: ', filename)
        raise Exception("Please check the length of error_tier: : ", filename)

      for i in range(len(txt_tier)):
        if txt_tier[i].name not in ['', ' ', '[SP]', '[U]'] and len(txt_tier[i].name) != 0 and txt_tier[i].name[0] != '*':
          if error_tier[i].name == '': # 没有错误
            error = 'W0'
          elif error_tier[i].name == '[1]': # 毫不相关的错误、嗯、啊语气词
            error = 'W1'
          elif error_tier[i].name == '[2]': # 声调错误、辅音缺失、元音替换等
            error = 'W2'
          new = pd.DataFrame({'Filename': filename, 'Task': task, 'Session': session,'Person': person, 'Start': txt_tier[i].start, 'End': txt_tier[i].stop, 'Text': txt_tier[i].name, 'Duration': txt_tier[i].stop - txt_tier[i].start, 'WrongInfo': error}, index=[1])

          self.Total_data = self.Total_data.append(new, ignore_index=True)
      self.Total_data = self.Total_data.sort_values(by=['Person', 'Filename'], ascending=True).reset_index(drop=True)
      # self.Total_data.to_csv(csv_path, encoding='utf_8_sig', index=False)

  def _get_grid_info(self, text_path):
    '''
      TODO:
        1. 获取所有信息，包括 文件名，task，session，person， start，end，text，duration
    '''
    file_data, select_data = [] , []
    logging.error('read file error {}'.format(text_path))
    ref_tier = tg.read_textgrid(text_path, "refphone")
    path = text_path.split('.')[0]
    wav_path = text_path.replace('.TextGrid', '.wav')
    if self._avi_frame_check(wav_path) > 15:
      logging.error("drop frams over 15 {}".format(wav_path))
      return file_data
    filename = os.path.basename(text_path)
    tmp = filename.split('.')[0].split('_')
    task = tmp[-3] + '_' + tmp[-2] + '_' + tmp[-1]
    person = tmp[0] + '_' + tmp[2] + '_' + tmp[1] if tmp[0] != 'repeat' else tmp[1] + '_' + tmp[3] + '_' + tmp[2]
    session = person + '_' + tmp[-3]
    
    count = 0
    tmp_list = []
    for entry in ref_tier:
      text = entry.name.strip()
      text = self._get_hanzi(text)
      if '' == text:
        if 0 < len(tmp_list) < 3:
          select_data.extend(tmp_list)
        tmp_list = []
        continue
      count += 1
      if text not in self.Syllable_Phone:
        logging.error("{} {}".format(text, text_path))
        continue
      vowel = self.Syllable_Vowel[text]
      syllable = self.Syllable_Phone[text]
      new = pd.DataFrame({'Filename': filename, 'Task': task, 'Session': session,'Person': person, 'Path':path, 'Start': entry.start, 'End': entry.stop, 'Text': text, 'Duration': entry.stop - entry.start, 'Syllable':syllable, 'Vowel':vowel, 'count':count}, index=[1])
      file_data.append(new)
      tmp_list.append(filename.spli('.')[0] + '_' + str(count))
    return {'all_seg':file_data, 'sel_seg':select_data}

  def _save_select_file(self, data, txt_file):
    with open(txt_file, 'w') as file:
      for item in data:
        file.write(str(item) + '\n')

  def text_csv(self, csv_path, multi_num=1):
    select_file_list = []
    if multi_num == 1:
      for text_path in self.text_files:
        file_result = self._get_grid_info(text_path)
        for new in file_result['all_seg']:
          if len(new) != 0:
            self.Total_data = self.Total_data.append(new, ignore_index=True)
    else:
      MP = MultiProcess()
      file_result_list = MP.multi_with_result(func=self._get_grid_info, arg_list=self.text_files, process_num=multi_num)
      for file_result in file_result_list:
        if not file_result:
          continue
        for new in file_result['all_seg']:
          if len(new) != 0:
            self.Total_data = self.Total_data.append(new, ignore_index=True)
        for select_file in file_result['sel_seg']:
          select_file_list.append(select_file)
    self.Total_data = self.Total_data.sort_values(by=['Person', 'Filename'], ascending=True).reset_index(drop=True)
    self.Total_data.to_csv(csv_path, encoding='utf_8_sig', index=False)
    # self._save_select_file(select_file_list, 'tmp/select_filename.txt')

def lsj_textgrid2csv():
  text_dir = "/mnt/shareEEx/liuxiaokang/data/MSDM/labeled_data/Control/N_10008_F"
  csv_path = "tmp/N_10008_F_allinfo.csv"
  T2C = Text2Csv()
  T2C.get_all_file(text_dir)
  T2C.lsj_text_csv()
  T2C.lsj_add_syllble(csv_path)

def textgrid2csv():
  text_dir = "/mnt/shareEEx/liuxiaokang/data/MSDM/labeled_data/20230605"
  csv_path = "tmp/20230803.csv"
  T2C = Text2Csv(text_dir)
  T2C.text_csv(csv_path, 1)

if __name__ == "__main__":
  textgrid2csv()