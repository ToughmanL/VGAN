#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	word_segment.py
 *@Time	: 2023-11-19 23:05:30
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''

import os
import csv
import pandas as pd

class WordSegement():
  def __init__(self):
    self.person_label = {}
    self.listofdict = []

  def read_label(self, label_csv):
    # read Lable 
    lfp = open(label_csv,'r',encoding='gbk')
    reader = csv.DictReader(lfp)
    for row in reader:
      Frenchay = int(row['Frenchay'])
      label_dict = {'Frenchay':row['Frenchay'],'fanshe':row['fanshe'], 'huxi':row['huxi'], 'chun':row['chun'], 'he':row['he'], 'ruane':row['ruane'], 'hou':row['hou'], 'she':row['she'], 'yanyu':row['yanyu']}
      self.person_label[row['Person']] = label_dict

  def read_scp_segments(self, scp_path):
    segment_path = os.path.join(scp_path, 'segment')
    wav_scp_path = os.path.join(scp_path, 'wav.scp')
    wav_name_path = {}
    with open(wav_scp_path, 'r') as fp:
      for line in fp:
        line = line.rstrip('\n')
        ll = line.split(' ')
        wav_name_path[ll[0]] = ll[1]
    with open(segment_path, 'r') as fp:
      for line in fp:
        line = line.rstrip('\n')
        ll = line.split(' ')
        if ll[0].split('_')[4] not in ['task1', 'task2', 'task3']:
          continue
        if float(ll[3])-float(ll[2]) < 0.025:
          continue
        wav_path = wav_name_path[ll[1]]
        preson = wav_path.split('/')[-2]
        tmp_dict = {'Person':preson, 'Path':wav_path, 'Start':ll[2], 'End':ll[3]}
        tmp_dict.update(self.person_label[preson])
        self.listofdict.append(tmp_dict)
  
  def data_balance(self, pd_data):
    persons = pd_data['Person'].unique()
    person_list = []
    for p in persons:
      target_len = 1000 if 'S' in p else 100
      person_data = pd_data[pd_data['Person'] == p]
      if len(person_data) > target_len:
        tmp_data = person_data.sample(n=target_len, random_state=1)
      else:
        tmp_data = person_data
        while len(tmp_data) < target_len:
          if target_len - len(tmp_data) > len(person_data):
            n_samp = len(person_data)
          else:
            n_samp = target_len - len(tmp_data)

          samp_date = person_data.sample(n=n_samp)
          tmp_data = pd.concat([tmp_data,samp_date])
      person_list.append(tmp_data)
    balance_data = pd.concat(person_list)
    return balance_data

  def write_csv(self, data_csv):
    result_data = pd.DataFrame(self.listofdict)
    balance_data = self.data_balance(result_data)
    balance_data.to_csv(data_csv, index=False)
  
  def count_data(self, data_csv):
    data = pd.read_csv(data_csv)
    listofdict = data.to_dict(orient='records')
    dur_list = []
    person_num = {}
    for dict in listofdict:
      dur = float(dict['End']) - float(dict['Start'])
      dur_list.append(dur)
      if dur < 0.025:
        print(dict)
      person = dict['Person']
      if person in person_num:
        person_num[person] += 1
      else:
        person_num[person] = 1
    print(person_num)
    dur_list.sort()
    for i in range(len(dur_list)):
      if i % int(len(dur_list)/10) == 0:
        print(dur_list[i])
    

if __name__ == '__main__':
  label_csv = '../data/Label.csv'
  scp_path = '/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-230422/examples/MSDM/230507/data/MSDM/'
  data_csv = '../data/result_intermediate/phase_setment_1119.csv'
  WS = WordSegement()
  WS.read_label(label_csv)
  WS.read_scp_segments(scp_path)
  WS.write_csv(data_csv)
  WS.count_data(data_csv)

  