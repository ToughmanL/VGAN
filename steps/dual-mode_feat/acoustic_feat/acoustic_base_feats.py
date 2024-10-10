#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 acoustic_feats.py
* @Time 	:	 2022/11/23
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 提取声学特征
'''
import os
import shutil
import pandas as pd
import numpy as np
import math
import parselmouth
from parselmouth.praat import call
from utils.multi_process import MultiProcess
from utils.get_files_dirs import FileDir
from local.get_gvme_feats import get_tmp_feats, read_gop_score, get_name2text
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',30)

class AcousticFeats():
  def __init__(self, gmm_seg_file, segment_dir, tmp_feat_dir, data_process_csv, gop_path):
    self.gmm_seg_file = gmm_seg_file
    self.wav_param_list = []
    self.FD = FileDir()
    self.gne_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'GNE')
    self.vfer_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'VFER')
    self.mfcc_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'mfcc')
    self.egemaps_dict = get_tmp_feats(tmp_feat_dir, segment_dir, 'egemaps')
    self.gop_dict = read_gop_score(gop_path)
    self.filename2text_dict = get_name2text(data_process_csv) 

  def _get_files(self, datadir, suff):
    path_list = []
    for root, dirs, files in os.walk(datadir, followlinks=True):
      for file in files:
        if file.split('.')[1] == suff:
          path_list.append(os.path.join(root, file))
    return path_list

  def _save_pitchdata(self, wav, pitch, st, et):
    txt_path = "tmp/pitch_count/" + wav.replace('.wav', '.txt')
    fst, fet = round(pitch.time_to_frame_number(st)), round(pitch.time_to_frame_number(et))
    data = pitch.to_array()[0][fst:fet]
    np.savetxt(txt_path, data, delimiter=',')

  def _get_st_et(self, pitch, wav, wav_path, st):
    '''
     @func _get_st_et
     @desc 找到元音pitch的开始结束时间，使用gmm切分作为基础
           找到praat预测的pitch段头尾，如果分段的话找最大段头尾
           如果分了多段，找第一段头，最后一段尾
     @param {}  
     @return {} 
    '''
    if st > pitch.duration/2:
      st = pitch.duration/10
    pitch_arr = pitch.to_array()[0]
    pitch_len = len(pitch_arr)
    vowel_beg = round(pitch.get_frame_number_from_time(st)+1)
    margin = math.floor((pitch_len-vowel_beg)/10)

    st_list = []
    et_list = []

    # 找到pitch段
    pitch_tmp = pitch_arr[vowel_beg+margin:pitch_len-margin].tolist()
    pitch_tmp = [ i[0] for i in pitch_tmp]
    pitch_tmp.insert(0, 0)
    pitch_tmp.append(0)
    for i in range(0, len(pitch_tmp)-1):
      if pitch_tmp[i] == 0 and pitch_tmp[i+1] != 0:
        st_list.append(i+vowel_beg+margin+1)
      if pitch_tmp[i] != 0 and pitch_tmp[i+1] == 0:
        et_list.append(i+vowel_beg+margin+1)
    if len(st_list) == 0:
      st_list.append(vowel_beg+margin)
      et_list.append(pitch_len-margin)

    # 找到最大段
    fst = st_list[0]
    fet = et_list[0]
    if len(st_list) > 1:
      for i in range(len(st_list)):
        if (et_list[i]-st_list[i]) > (fet-fst):
          fet = et_list[i]
          fst = st_list[i]
      # 最大段如果小于5帧则使用整段
      if et_list[i]-st_list[i] < 5:
        fst = st_list[0]
        fet = et_list[-1]

    ft = pitch.get_time_from_frame_number(fst+1)
    et =pitch.get_time_from_frame_number(fet+1)
    # print(pitch.duration-st, et-ft)
    if et-ft < 0:
      print({'vowel_beg':vowel_beg, 'pitch_len':pitch_len, 'st_list':st_list, 'et_list':et_list})
    return ft, et

  def _get_wavs_start(self, wav_dir):
    '''
      找到每条音频的 名称、路径、人名、元音起始点
    '''
    wav_path_list = self._get_files(wav_dir, 'wav')
    voiced_data = pd.read_csv(self.gmm_seg_file)
    for wav_path in wav_path_list:
      wav = os.path.basename(wav_path)
      if wav not in voiced_data['Filename'].unique():
        continue
      vowel_st = voiced_data[voiced_data['Filename'] == wav].reset_index(drop=True).loc[0, 'start']
      person = voiced_data[voiced_data['Filename'] == wav].reset_index(drop=True).loc[0, 'Person']
      self.wav_param_list.append({'wav':wav, 'wav_path':wav_path, 'person':person, 'start':vowel_st})
      # break
    
  def _kmeans(self, Format, st, et):
    '''
     @func _kmeans
     @desc 使用kmeans对f1,f2,f3进行聚类使其预测结果更合理，然而并没有用
     @param {Format数据，开始结束时间}  
     @return {F1, F2, F3的均值和方差} 
    '''
    F_data = []
    for point in range(int(Format.get_frame_number_from_time(st)+1), int(Format.get_frame_number_from_time(et)+1)):
      point = 1 if point < 1 else point
      t = Format.frame_number_to_time(point)
      f1 = call(Format, "Get value at time", 1, t, 'Hertz', 'Linear')
      f2 = call(Format, "Get value at time", 2, t, 'Hertz', 'Linear')
      f3 = call(Format, "Get value at time", 3, t, 'Hertz', 'Linear')
      if not np.isnan(f1):
        F_data.append([f1, f1])
      if not np.isnan(f2):
        F_data.append([f2, f2])
      if not np.isnan(f3):
        F_data.append([f3, f3])
      kmeans = KMeans(n_clusters=3, random_state=0, max_iter=100).fit(np.array(F_data))
    F_list_dict = {'lab_0':[], 'lab_1':[], 'lab_2':[]}
    for i in range(len(kmeans.labels_)):
      if kmeans.labels_[i] == 0:
        F_list_dict['lab_0'].append(F_data[i][0])
      elif kmeans.labels_[i] == 1:
        F_list_dict['lab_1'].append(F_data[i][0])
      elif kmeans.labels_[i] == 2:
        F_list_dict['lab_2'].append(F_data[i][0])
    mean_std = []
    for key, value in F_list_dict.items():
      mean_std.append({'means':np.mean(np.array(value)), 'std':np.std(np.array(value))})

    sort_d = sorted(mean_std, key= lambda i:i['means'])
    meanF1, stdF1, meanF2, stdF2, meanF3, stdF3 = sort_d[0]['means'], sort_d[0]['std'], sort_d[1]['means'],sort_d[1]['std'],sort_d[2]['means'],sort_d[2]['std']

    return meanF1, stdF1, meanF2, stdF2, meanF3, stdF3

  def _get_voiced_feats(self, wav_param):
    '''
    获取pitch_mean/std、Intensity_mean/std、F1_mean/std、
       F2_mean/std、F3_mean/std、Syllable_dur、Vowel_dur
    '''
    wav_path, wav, person, gmm_st = wav_param['wav_path'], wav_param['wav'], wav_param['person'], wav_param['start']
    tmp = wav.split('.')[0].split('_')  # infomation
    text = self.filename2text_dict[wav.split('.')[0]]

    # Step1 找到pitch起始时刻
    sound = parselmouth.Sound(wav_path)
    try:
      pitch = sound.to_pitch_ac(None, 100.0, 20, False, 0.01, 0.5, 0.01, 0.35, 0.14, 335.0)
    except:
      return {'Filename': wav.split('.')[0], 'Person': person, 'TEXT': text, 'Pitch_mean': None, 'Pitch_sd': None, 'Intensity_mean': None, 'Intensity_sd': None, 'F1_mean': None, 'F1_sd': None, 'F2_mean': None, 'F2_sd': None, 'F3_mean': None, 'F3_sd': None, 'Syllable_dur': None, 'Vowel_dur': None, 'Jitter': None, 'Shimmer': None, 'HNR': None, 'vowel_st':None, 'vowel_et':None, 'gmm_st':gmm_st}
    st, et = self._get_st_et(pitch, wav, wav_path, gmm_st)
    
    # pitch = sound.to_pitch()
    duration = pitch.duration
    vowel_dur = et - st

    # 保存pitch 数据
    self._save_pitchdata(wav, pitch, st, et)

    # 获取80%稳定段 F0_mean, std
    meanF0 = call(pitch, "Get mean", st, et, "Hertz")
    stdF0 = call(pitch, "Get standard deviation", st, et, "Hertz")

    # 获取80%稳定段 Intensity_mean, std
    try:
      Intensity = sound.to_intensity()
    except:
      return {'Filename': wav.split('.')[0], 'Person': person, 'TEXT': text, 'Pitch_mean': None, 'Pitch_sd': None, 'Intensity_mean': None, 'Intensity_sd': None, 'F1_mean': None, 'F1_sd': None, 'F2_mean': None, 'F2_sd': None, 'F3_mean': None, 'F3_sd': None, 'Syllable_dur': None, 'Vowel_dur': None, 'Jitter': None, 'Shimmer': None, 'HNR': None, 'vowel_st':st, 'vowel_et':et, 'gmm_st':gmm_st}
    meanInt = call(Intensity, "Get mean", st, et, "dB")
    stdInt = call(Intensity, "Get standard deviation", st, et)

    # 获取80%稳定段 F1/2/3
    # Format = sound.to_formant_burg()
    Format = call(sound, "To Formant (burg)", 0.0025, 5, 4000, 0.025, 30)
    meanF1 = call(Format, "Get mean", 1, st, et, "Hertz")
    stdF1 = call(Format, "Get standard deviation", 1, st, et, "Hertz")
    meanF2 = call(Format, "Get mean", 2, st, et, "Hertz")
    stdF2 = call(Format, "Get standard deviation", 2, st, et, "Hertz")
    meanF3 = call(Format, "Get mean", 3, st, et, "Hertz")
    stdF3 = call(Format, "Get standard deviation", 3, st, et, "Hertz")

    # 获取80%元音稳定段 Jitter Shimmer HNR
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    Jitter = call(pointProcess, "Get jitter (local)", st, et, 0.0001, 0.02, 1.3)
    Shimmer = call([sound, pointProcess], "Get shimmer (local)", st, et, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR = call(harmonicity, "Get mean", st, et)

    result_dict = {'Filename': wav.split('.')[0], 'Person': person, 'TEXT': text, 'Pitch_mean': meanF0, 'Pitch_sd': stdF0, 'Intensity_mean': meanInt, 'Intensity_sd': stdInt, 'F1_mean': meanF1, 'F1_sd': stdF1, 'F2_mean': meanF2, 'F2_sd': stdF2, 'F3_mean': meanF3, 'F3_sd': stdF3, 'Syllable_dur': duration, 'Vowel_dur': vowel_dur, 'Jitter': Jitter, 'Shimmer': Shimmer, 'HNR': HNR, 'vowel_st':st, 'vowel_et':et, 'gmm_st':gmm_st}
    return result_dict

  def _get_base_feats(self, wav_param):
    result_dict = self._get_voiced_feats(wav_param)
    filename = result_dict['Filename']
    gne_err_no, vfer_err_no = 0, 0
    if filename in self.gne_dict:
      gne = self.gne_dict[filename]
    else:
      gne = None
      print(filename, 'GNE None')
    if filename in self.vfer_dict:
      vfer = self.vfer_dict[filename]
    else:
      vfer = None
      print(filename, 'VFER None')
    result_dict.update({'gne':gne, 'vfer':vfer})
    if filename in self.gop_dict:
      gop = self.gop_dict[filename]
    else:
      gop = [None, None]
      print(filename, 'GOP None')
    result_dict.update({'gop_con':gop[0], 'gop_vow':gop[1]})
    if filename in self.mfcc_dict:
      mfcc = self.mfcc_dict[filename]['featpath']
    else:
      mfcc = None
      print(filename, 'mfcc None')
    result_dict.update({'mfcc':mfcc})
    if filename in self.egemaps_dict:
      egemaps = self.egemaps_dict[filename]['featpath']
    else:
      egemaps = None
      print(filename, 'egemaps None')
    result_dict.update({'egemaps':egemaps})
    return result_dict

  def interface_acoustic(self, wav_dir, csv_path, multi_num=1):
    self._get_wavs_start(wav_dir)

    results = []
    if multi_num > 1:
      MP = MultiProcess()
      results = MP.multi_with_result(func=self._get_base_feats, \
                                    arg_list=self.wav_param_list, process_num=multi_num)
    else:
      for wav_param in self.wav_param_list:
        res = self._get_base_feats(wav_param)
        results.append(res)
    
    df = pd.DataFrame(results)
    df.dropna(axis=0, how='any', inplace=True)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
  segment_dir = 'data/segment_data'
  tmp_feat_dir = 'tmp/gvmeg'
  gmm_seg_file = 'data/result_intermediate/230617_gmm_con_vowel_segment.csv'
  data_process_csv = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/20230921.csv'
  gop_path = 'data/segment_data/kaldidata/data/gop.ark'

  wav_dir='data/segment_data/Patient/S_00037_M/'
  csv_path= 'tmp/pitch_ac-acousticfeat_37.csv'
  
  AF = AcousticFeats(gmm_seg_file, segment_dir, tmp_feat_dir, data_process_csv, gop_path)
  AF.interface_acoustic(wav_dir, csv_path, multi_num=60)
