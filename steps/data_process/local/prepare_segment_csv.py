#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
import os, csv, sys

class PrepareSegmentCsv():
  def __init__(self, scpfile, label_csv) -> None:
    self.person_label = self.read_label(label_csv)
    self.id_list = self.read_scp_segments(scpfile)
  
  def read_label(self, label_csv='data/Label.csv'):
    label = pd.read_csv(label_csv, encoding='gbk')
    label_dict = label.set_index('Person').T.to_dict()
    return label_dict
  
  def read_scp_segments(self, scp_path):
    id_list = []
    with open(scp_path, 'r') as fp:
      for line in fp:
        line = line.rstrip('\n').split(' ')
        id_list.append(line[0])
    return id_list
  
  def id2person(self, id):
    item = id.replace('repeat_', '')
    nf_num = item.split('_')[0:3]
    person = nf_num[0] + '_' + nf_num[2] + '_' + nf_num[1]
    return person
  
  def write_segcsv(self, seg_csv):
    with open(seg_csv, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      labels_list = ['Frenchay', 'fanshe', 'huxi', 'chun', 'he', 'ruane', 'hou', 'she', 'yanyu']
      writer.writerow(['Person', 'Segname'] + labels_list)
      for id in self.id_list:
        person = self.id2person(id)
        label_score = []
        for label in labels_list:
          label_score.append(self.person_label[person][label])
        writer.writerow([person, id] + label_score)
    print(f"Write segment csv to {seg_csv}")


if __name__ == "__main__":
  scpfile = "/mnt/shareEEx/liuxiaokang/workspace/av-dysarthria-diagnosis/egs/msdm/data/MSDM/text"
  label_file = "data/Label.csv"
  seg_csv = "tmp/segment.csv"
  psc = PrepareSegmentCsv(scpfile, label_file)
  psc.write_segcsv(seg_csv)


    
  


