#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   segment_audio_video.py
@Time    :   2023/06/09 23:24:23
@Author  :   lxk 
@Version :   2.0
@Contact :   xk.liu@siat.ac.cn
@License :   (C)Copyright 2022-2025, lxk&AISML
@Desc    :   None
'''

import os
import pandas as pd
import numpy as np
import cv2
from utils.multi_process import MultiProcess 
import warnings
warnings.filterwarnings('ignore')


class SegmentAudioVideo():
  def __init__(self, video_path, out_put_path):
    self.video_path = video_path
    self.out_put_path = out_put_path
  
  def _lsj_segment_file(self, single_data): # 音视频切分
    person = single_data['Person']
    categ = 'Patient' if person[0] == 'S' else 'Control'
    
    # 创建目录
    output_person_path = os.path.join(self.out_put_path, categ, person)
    if not os.path.exists(output_person_path):
      os.makedirs(output_person_path)

    video_path = self.video_path + '/' + categ + '/' + person + '/' + single_data['Filename'].split('.')[0] + '.avi'
    cap = cv2.VideoCapture(video_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))  # 帧速率
    frame_num = cap.get(7)
    video_duration = frame_num / video_fps
    audio_path = self.video_path + '/' + categ + '/' + person + '/' + single_data['Filename'].split('.')[0] + '.wav'

    count = str(single_data['count'])
    text = single_data['Text'].replace(' ', '')
    wrong_info = 'Wrong' + single_data['WrongInfo'][-1]
    audio_start = round(single_data['Start'], 3)
    audio_duration = round(single_data['End'] - audio_start, 3)

    video_start = round(single_data['Start'], 3)
    video_start = 0 if video_start < 0 else video_start

    video_end = round(single_data['End'], 3)
    video_end = video_duration if video_end > video_duration else video_end

    # 输出wav和avi
    output_audio_path = os.path.join(output_person_path, audio_path.split('/')[-1].split('.')[0])+'_'+text+'_'+count+'_'+wrong_info+'.wav'
    output_video_path = os.path.join(output_person_path, video_path.split('/')[-1].split('.')[0])+'_'+text+'_'+count+'_'+wrong_info+'.avi'

    # 切分segment
    conson_audio_command = 'sox ' + audio_path + ' ' + output_audio_path + ' trim ' + str(audio_start) + ' ' + str(audio_duration)
    os.system(conson_audio_command)
    conson_video_command = 'ffmpeg -y -i ' + video_path + ' -ss ' + str(video_start) + ' -to ' + str(video_end) + ' -c copy ' + output_video_path
    os.system(conson_video_command)
    print(audio_path.split('/')[-1].split('.')[0]+'_'+text+'_'+count+'_'+wrong_info)

  def _segment_file(self, single_data):
    ori_path = single_data['Path']
    categ = 'Patient' if single_data['Person'][0] == 'S' else 'Control'
    out_dir = os.path.join(self.out_put_path, categ)
    output_path = out_dir + '/' + ori_path.split(categ)[-1]
    # 获取音视频地址
    count = str(single_data['count'])
    ori_avi_path = ori_path + '.avi'
    ori_wav_path = ori_path + '.wav'
    output_avi_path = output_path + '_' + count + '.avi'
    output_wav_path = output_path + '_' + count + '.wav'
    # 获取视频时间
    cap = cv2.VideoCapture(ori_avi_path)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))  # 帧速率
    frame_num = cap.get(7)
    video_duration = frame_num / video_fps
    video_start = round(single_data['Start'], 3)
    video_start = 0 if video_start < 0 else video_start
    video_end = round(single_data['End'], 3)
    video_end = video_duration if video_end > video_duration else video_end
    # 获取音频时间
    text = single_data['Text'].replace(' ', '')
    audio_start = round(single_data['Start'], 3)
    audio_duration = round(single_data['End'] - audio_start, 3)
    # 切分数据
    conson_audio_command = 'sox ' + ori_wav_path + ' ' + output_wav_path + ' trim ' + str(audio_start) + ' ' + str(audio_duration)
    conson_video_command = 'ffmpeg -loglevel quiet -y -i ' + ori_avi_path + ' -ss ' + str(video_start) + ' -to ' + str(video_end) + ' -c copy ' + output_avi_path
    os.system(conson_audio_command)
    os.system(conson_video_command)

  def seg_data(self, csv_path, multi=1): # 获取所有数据
    all_data = []
    wav_time_info = pd.read_csv(csv_path)
    all_data = wav_time_info.T.to_dict().values()

    for single_data in all_data:
      ori_path = single_data['Path']
      categ = 'Patient' if single_data['Person'][0] == 'S' else 'Control'
      out_dir = os.path.join(self.out_put_path, categ, single_data['Person'])
      if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 是否使用多线程
    if multi == 1:
      for single_data in all_data:
        self._segment_file(single_data)
    else:
      MP = MultiProcess()
      MP.multi_not_result(func=self._segment_file, arg_list=all_data)

  
if __name__ == "__main__":
  video_path = "/mnt/shareEx/lushangjun/Lipspeech/database/consonant_vowel_all_0905/audio_vedio"
  out_put_path = "data/segment_data/"
  csv_path = "./tmp/20230219.csv"
  SAV = SegmentAudioVideo(video_path, out_put_path)
  SAV.get_all_data(csv_path, 50)