#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	sel_syll_word.py
 *@Time	: 2023-08-03 00:04:07
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''

import pandas as pd

class SelectSegment:
  def __init__(self, sel_path, csv_path) -> None:
    self.feat_data = []
    self.new_feat_data = []
    self.sel_filename_set = set()
    self._read_csv(csv_path)
    self._read_txt(sel_path)
  
  def _read_txt(self, txt_path):
    with open(txt_path, 'r') as fp:
      for line in fp:
        self.sel_filename_set.add(line.rstrip('\n'))

  def _read_csv(self, csv_path):
    df = pd.read_csv(csv_path)
    self.feat_data = df.T.to_dict().values()

  def save_csv(self, csv_path):
    df = pd.DataFrame(self.new_feat_data)
    df.to_csv(csv_path, index=False)

  def select_syll_word(self):
    # 找到每个task最大的count数
    for segm in self.feat_data:
      filename = segm['Filename']
      if filename in self.sel_filename_set:
        self.new_feat_data.append(segm)

if __name__ == "__main__":
  sel_path = 'tmp//select_filename.txt'
  sour_csv_path = 'tmp/pitch_ac-acousticfeat_0625.csv'
  dest_csv_path = 'pitch_ac-acousticfeat_0803.csv'
  SS = SelectSegment(sel_path, sour_csv_path)
  SS.select_syll_word()
  SS.save_csv(dest_csv_path)