#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2022/10/31 23:40:00
@Author  :   lxk 
@Version :   1.0
@Contact :   xk.liu@siat.ac.cn
@License :   (C)Copyright 2022-2025, lxk&AISML
@Desc    :   None
'''

import os
import pandas as pd


class Data_Process():
  def __init__(self) -> None:
    pass

  def textgrid_csv(self):
    from local.textgrid_csv import Text2Csv
    # read textgrid files and convert to csvfile
    text_dir = "/mnt/shareEEx/liuxiaokang/data/MSDM/labeled_data/20230605"
    csv_path = "tmp/20230921.csv"
    T2C = Text2Csv(text_dir)
    T2C.text_csv(csv_path, 80)
    print("data_process.py textgrid_csv finished")

  def segment_audio_video(self):
    from local.segment_audio_video import SegmentAudioVideo
    # audio and video segmentation
    video_path = "/mnt/shareEEx/liuxiaokang/data/MSDM/labeled_data/20230605"
    out_put_path = "./tmp/230617_segmen_data"
    csv_path = "./tmp/20230617.csv"
    SAV = SegmentAudioVideo(video_path, out_put_path)
    SAV.seg_data(csv_path, 50)
    print("data_process.py segment_audio_video finished")

  def merge_csv(self, *csvs, dest_csv):
    # merge csvs
    all_csv_datas = [pd.read_csv(f) for f in csvs]
    df_merged = pd.concat(all_csv_datas, ignore_index=True)
    df_merged = df_merged.reset_index(drop=True)
    df_merged.to_csv(dest_csv,index=False)

  def gmm_div_vow_con(self):
    from GMM.feats import DataPrep
    from GMM.gmm import WriteGMMResult
    multi_num = 50
    withflag = False
    # features extraction
    wavdir = "/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/230617_segmen_data/Control" # for implement
    implement_feat_path = "data/gmmfeat/Control"
    DP = DataPrep(wavdir, implement_feat_path)
    DP.multi_process(withflag, multi_num)

    # gmm split
    featdir = "data/gmmfeat/"
    modeldir = "data/model/"
    cons_gmm_path = modeldir + "gmm_NS_conso_com100_max20_covfull.smn"
    vowel_gmm_path = modeldir + "gmm_NS_vowel_com80_max20_covfull.smn"
    result_path = "result/230617_gmm_con_vowel_segment.csv"
    WG = WriteGMMResult(multi_num)
    WG.writeresult(featdir, cons_gmm_path, vowel_gmm_path, result_path)

  def data_check(self, seg_dir):
    # check data
    avi_list = []
    wav_list = []
    for root, dirs, files in os.walk(seg_dir):
      for file in files:
        if '.avi' in file:
          avi_list.append(file.split('.')[0])
        if '.wav' in file:
          wav_list.append(file.split('.')[0])
    if avi_list.sort() != wav_list.sort():
      print("data_check error")
    else:
      print("data_check success")
    print("data_process.py data_check finished")

if __name__ == "__main__":
  DP = Data_Process()
  DP.textgrid_csv()
  # DP.segment_audio_video()
  # DP.merge_csv('./tmp/Mono_All_data.csv', './tmp/N_10008_F_allinfo.csv', dest_csv="./tmp/Apersons_Ainfos.csv")
  # seg_dir = 'tmp/230617_segmen_data'
  # DP.data_check(seg_dir)
  # DP.gmm_div_vow_con()

