
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# 从textgrid文件中提取每个音节对应的发音时间（静止、开始、元音、结束）

import os
import pandas as pd
from util import textgrid as tg
from util.syllable_process import SyllableProcess
from util.multi_process import MultiProcess 
from util.syllable_feats import DataPrep
from util.gmm import WriteGMMResult
import warnings
warnings.filterwarnings("ignore")

class TextgridProcess():
  def __init__(self, Syllable_path, DATA_PATH, SAVE_PATH, multi_num1, multi_num2, cons_gmm_path, vowel_gmm_path):
    self.EMPTY = [""," "]
    self.DATA_PATH = DATA_PATH
    self.SAVE_PATH = SAVE_PATH
    self.syllable_process = SyllableProcess(Syllable_path)
    self.mono_vowel_dic = self.syllable_process.Get_All_Syllable()
    self.mono_vowel_text = list(self.mono_vowel_dic.keys())
    # self.mono_vowel_syllable = list(self.mono_vowel_dic.values())
    self.all_refphone_tier = []
    self.all_tg_path = []
    self.Total_data = pd.DataFrame(None, columns=['filepath', 'silence_end','syllable_start', 'vowel_start', 'syllable_end','text','syllable','count', 'type'])
    self.multi_num1 = multi_num1
    self.multi_num2 = multi_num2
    self.DP = DataPrep()
    self.WG = WriteGMMResult()
    self.cons_gmm_path = cons_gmm_path
    self.vowel_gmm_path = vowel_gmm_path

  #读取所有的textgrid路径
  def _get_tg_path(self):
    for root, dirs, files in os.walk(self.DATA_PATH, followlinks=True):
      for file in files:
        if file.endswith(".TextGrid") and (file.startswith("N") or file.startswith("S") ):
          self.all_tg_path.append(os.path.join(root, file))

  # 处理一个TextGrid文件,返回对应的refphone_tier字典数据
  def _ReadTextgrid(self, TextGrid_path):
    Filepath = TextGrid_path.split(".")[0]+".wav"
    tgrid = tg.read_textgrid(TextGrid_path, 'refphone')
    refphone_tier = [Filepath, tgrid]
    return refphone_tier
  
  #找到一个segment
  def _find_seg(self, length, refphone_tier, i):
    tmp_syllable = [] 
    j = i
    while j < length and refphone_tier[j].name not in self.EMPTY:
      if refphone_tier[j].name in self.mono_vowel_text:
        text = refphone_tier[j].name
        syllable = self.mono_vowel_dic[text]
        tmp_syllable.append([refphone_tier[j].start, refphone_tier[j].stop, text, syllable])
      j += 1
    return j, tmp_syllable
  
  #将textgrid中的segment(syllable、words)写入
  def _Write_syllable(self, tmp_data, Count, Filepath, Silence_end, tmp_syllable):
    t = "0" if len(tmp_syllable) == 1 else "1" #type共2种，1为短语,0为单字
    for i in range(len(tmp_syllable)):
      Count += 1
      Syllable_start = tmp_syllable[i][0]
      Syllable_end = tmp_syllable[i][1]
      Text = tmp_syllable[i][2]
      Syllable = tmp_syllable[i][3]
      new = pd.DataFrame({'filepath':Filepath, 'silence_end':Silence_end,\
                          'syllable_start':Syllable_start, 'vowel_start':None, 'syllable_end':Syllable_end, \
                          'text':Text,'syllable':Syllable,'count':Count, 'type': t}, index=[1])
      tmp_data = tmp_data.append(new, ignore_index=True)
    return tmp_data, Count
 
  def _Process_refphone(self, *arg):
    Filepath = arg[0][0]
    refphone_tier = arg[0][1]
    file_name=Filepath.split("/")[-1]
    print("dealing:",file_name)
    tmp_data = pd.DataFrame(columns=['filepath', 'silence_end','syllable_start', 'vowel_start', 'syllable_end','text','syllable','count', 'type'])
    Count = 0
    i = 1
    length = len(refphone_tier) - 1
    while i < length and i > 0:
      # Cur_Syllable = refphone_tier[i]
      Before_Syllable = refphone_tier[i-1]
      Before_text = Before_Syllable.name   
      if Before_text in self.EMPTY:
        i, tmp_syllable = self._find_seg(length, refphone_tier, i)
        #判断是否是syllable、words（2）
        seg_len = len(tmp_syllable)
        if seg_len == 1 or seg_len == 2:
          tmp_data, Count = self._Write_syllable(tmp_data, Count, Filepath, refphone_tier[0].stop, tmp_syllable)
      i+=1    
    #提取syllable和words的发音特征npy
    if len(tmp_data) > 0:
      one_wav_feats = self.DP.feats_process(tmp_data[["filepath", "syllable_start", "syllable_end"]].values)
      fin_data = self.WG.writeresult(self.cons_gmm_path, self.vowel_gmm_path, one_wav_feats)
      tmp_data['vowel_start'] = fin_data
    return tmp_data

  #外部调用接口
  def multi_process(self):
    #读取所有textgrid路径
    self._get_tg_path()
    MP = MultiProcess()

    # 读取tg文件是否使用多线程,共5288条数据
    if self.multi_num1 == 1:
      for TextGrid_path in self.all_tg_path:
        refphone_tier = self._ReadTextgrid(TextGrid_path)
        self.all_refphone_tier.append(refphone_tier)
    else:
      self.all_refphone_tier = MP.multi_with_result(func=self._ReadTextgrid, arg_list=self.all_tg_path, process_num=self.multi_num1)

    #处理tg文件是否使用多线程
    if self.multi_num2 == 1:
      for entry in self.all_refphone_tier:
        res = self._Process_refphone(entry)
        self.Total_data = self.Total_data.append(res, ignore_index=True)
    else:
      res = MP.multi_with_result(func=self._Process_refphone, arg_list=self.all_refphone_tier, process_num=self.multi_num2)
      for re in res:
        self.Total_data = self.Total_data.append(re, ignore_index=True)

    self.Total_data = self.Total_data.sort_values(by=['filepath'], ascending=True).reset_index(drop=True)
    self.Total_data.to_csv(self.SAVE_PATH, encoding='utf_8_sig', index=False) 

if __name__ ==  "__main__":
  Syllable_path="/mnt/shareEEx/caodi/workspace/code/video_only/info/Mono_Vowel_Syllables.txt"
  DATA_PATH = "/mnt/shareEEx/caodi/workspace/data/20230605"
  SAVE_PATH = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/extract_syllable_time/result/230816_segment_time.csv"

  multi_num1 = 10
  multi_num2 = 15

  modeldir = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/extract_syllable_time/model/"
  cons_gmm_path = modeldir + "gmm_NS_conso_com100_max20_covfull.smn"
  vowel_gmm_path = modeldir + "gmm_NS_vowel_com80_max20_covfull.smn"

  
  test = TextgridProcess(Syllable_path, DATA_PATH, SAVE_PATH, multi_num1, multi_num2, cons_gmm_path, vowel_gmm_path)
  test.multi_process()
  
   
  

  


