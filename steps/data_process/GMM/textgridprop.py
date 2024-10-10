#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
#txt——dict中少了er2和er4，一共60个音节
txt_dict = {'八':'ba1', '波':'bo1', '脖':'bo2', '逼':'bi1','趴':'pa1', '爬':'pa2', '泼':'po1', '批':'pi1',
        '扑':'pu1', '妈':'ma1', '摸':'mo1', '模':'mo2', '法':'fa3', '德':'de2', '低':'di1', '读':'du2', '他':'ta1', '特':'te4',
        '踢':'ti1', '凸':'tu1', '努':'nu3', '女':'nv3', '拉':'la1', '乐':'le4', '李':'li3', '吕':'lv3', '姑':'gu1', '咔':'ka1',
        '科':'ke1', '哭':'ku1', '喝':'he1', '呼':'hu1', '居':'jv1', '七':'qi1', '区':'qv1', '需':'xv1', '渣':'zha1', '猪':'zhu1',
        '车':'che1', '吃':'chi1', '出':'chu1', '奢':'she1', '十':'shi2', '书':'shu1', '热':'re4', '则':'ze2', '字':'zi4', '租':'zu1',
        '次':'ci4', '粗':'cu1', '色':'se4', '四':'si4', '阿':'a1', '瓦':'wa3', '一':'yi1', '五':'wu3', '迂':'yv1', '故': 'gu4'}

real_dict = {'ba1':['b','a','1'], 'bo1':['b','o','1'], 'bo2':['b','o','2'], 'bi1':['b','i','1'], 'er2':['None','er','2'],
            'er4':['None','er','4'], 'pa1':['p','a','1'], 'pa2':['p','a','2'], 'po1':['p','o','1'], 'pi1':['p','i','1'],
            'pu1':['p','u','1'], 'ma1':['m','a','1'], 'mo1':['m','o','1'], 'mo2':['m','o','2'], 'fa3':['f','a','3'],
            'de2':['d','e','2'], 'di1':['d','i','1'], 'du2':['d','u','2'], 'ta1':['t','a','1'], 'te4':['t','e','4'],
            'ti1':['t','i','1'], 'tu1':['t','u','1'], 'nu3':['n','u','3'], 'nv3':['n','v','3'], 'la1':['l','a','1'],
            'le4':['l','e','4'], 'li3':['l','i','3'], 'lv3':['l','v','3'], 'gu1':['g','u','1'], 'ka1':['k','a','1'],
            'ke1':['k','e','1'], 'ku1':['k','u','1'], 'he1':['h','e','1'], 'hu1':['h','u','1'], 'jv1':['j','v','1'],
            'qi1':['q','i','1'], 'qv1':['q','v','1'], 'xv1':['x','v','1'], 'zha1':['zh','a','1'], 'zhu1':['zh','u','1'],
            'che1':['ch','e','1'], 'chi1':['ch','i','1'], 'chu1':['ch','u','1'], 'she1':['sh','e','1'], 'shi2':['sh','i','2'],
            'shu1':['sh','u','1'], 're4':['r','e','4'], 'ze2':['z','e','2'], 'zi4':['z','i','4'], 'zu1':['z','u','1'],
            'ci4':['c','i','4'], 'cu1':['c','u','1'], 'se4':['s','e','4'], 'si4':['s','i','4'], 'a1':['None','a','1'],
            'wa3':['w','a','3'], 'yi1':['None','yi','1'], 'wu3':['None','wu','3'], 'yv1':['None','yv','1'], 'gu4':['g', 'u', '4']}

Vowel_dict = {'ba1':'a', 'bo1':'o', 'bo2':'o', 'bi1':'i', 'pa1':'a', 'pa2':'a', 'po1':'o', 'pi1':'i', 'pu1':'u',
              'ma1':'a', 'mo1':'o', 'mo2':'o', 'fa3':'a', 'de2':'e', 'di1':'i', 'du2':'u', 'ta1':'a', 'te4':'e',
              'ti1':'i', 'tu1':'u', 'nu3':'u', 'nv3':'v', 'la1':'a', 'le4':'e', 'li3':'i', 'lv3':'v', 'gu1':'u',
              'ka1':'a', 'ke1':'e', 'ku1':'u', 'he1':'e', 'hu1':'u', 'jv1':'v', 'qi1':'i', 'qv1':'v', 'xv1':'v',
              'zha1':'a', 'zhu1':'u', 'che1':'e', 'chi1':'i', 'chu1':'u', 'she1':'e', 'shi2':'i', 'shu1':'u', 're4':'e',
              'ze2':'e', 'zi4':'i', 'zu1':'u', 'ci4':'i', 'cu1':'u', 'se4':'e', 'si4':'i', 'a1':'a', 'wa3':'a',
              'yi1':'i', 'wu3':'u', 'yv1':'v', 'er2': 'er', 'er4': 'er', 'gu4':'u'}

whole_syllable = ['a1', 'wa3', 'yi1', 'wu3', 'yv1', 'er2', 'er4']

class TextGridProcess():
  def __init__(self, textpath):
    self.textpath = textpath
    self.all_data = None

  def read_text(self):
    self.all_data = pd.read_csv(self.textpath, delimiter='\t', header=0)
    print(self.all_data.head)
    self.all_data = self.all_data.fillna('None')
    self.all_data['错误方式'] = '正确'
    self.all_data['Person'] = None
    self.all_data['Vowel'] = None
    for i in range(len(self.all_data)):
      self.all_data.loc[i, 'Person'] = self.all_data.loc[i, '文件名'].split('/')[-2]
      text = self.all_data.loc[i, 'TEXT']
      if text == 'None' \
         or text not in txt_dict \
         or txt_dict[text] in whole_syllable:
        self.all_data.loc[i, '错误方式'] = 'TEXT不在范围内'
        continue
      if self.all_data.loc[i, '音节调类'] == 'None':
        self.all_data.loc[i, '错误方式'] = '音调缺失'
        continue
      hpy_text = self.all_data.loc[i, '音节'] + str(int(self.all_data.loc[i, '音节调类']))
      hpy_text = hpy_text.replace(' ', '')
      if hpy_text != txt_dict[text]:
        # 以下对错误的音分类
        text = txt_dict[self.all_data.loc[i,'TEXT']]
        sm = self.all_data.loc[i,'声母'];ym = self.all_data.loc[i,'韵母'];sd = str(self.all_data.loc[i,'音节调类'])
        sm_real = real_dict[text][0];ym_real = real_dict[text][1];sd_real = real_dict[text][2]

        # 对每个错误音频进行分类
        if text not in whole_syllable:
          # 对非整体认读音节进行处理
          if (sm == sm_real) and (ym == ym_real) and (sd != sd_real):
              self.all_data.loc[i, '错误方式'] = '声调错误'
          elif (sm != 'None') and (sm != sm_real) and (ym == ym_real):
              self.all_data.loc[i, '错误方式'] = '辅音替换'
          elif (sm == 'None') and (ym == ym_real):
              self.all_data.loc[i, '错误方式'] = '辅音缺失'
          elif (sm == sm_real) and  (ym != ym_real):
              self.all_data.loc[i, '错误方式'] = '元音替换'
          elif (sm != 'None') and (sm != sm_real) and (ym != ym_real):
              self.all_data.loc[i, '错误方式'] = '有意义音节替换'
          elif (sm == 'None') and (ym not in ['e','eng','en','a']):
              self.all_data.loc[i, '错误方式'] = '有意义音节替换'
          elif (sm == 'None') and (ym in ['e','eng','en','a']):
              self.all_data.loc[i, '错误方式'] = '无意义音节替换'
        else:
          # 对整体认读音节进行处理
          if (sm == 'None') and (ym == ym_real) and (sd != sd_real):
              self.all_data.loc[i, '错误方式'] = '声调错误'
          elif (sm == 'None') and (ym != ym_real) and (ym not in ['e','eng','en','a']):
              self.all_data.loc[i, '错误方式'] = '有意义音节替换'
          elif (sm != 'None') and (ym != ym_real) :
              self.all_data.loc[i, '错误方式'] = '有意义音节替换'
          elif (sm == 'None') and (ym != ym_real) and (ym in ['e','eng','en','a']):
              self.all_data.loc[i, '错误方式'] = '无意义音节替换'

  def delete_error(self):
    # 剔除正常人错误发音，病人整体错误发音
    normal_index = []
    patient_index = []
    for i in range(len(self.all_data)):
      person = self.all_data.loc[i, '文件名'].split('/')[-2]
      if person not in ["S_00011_M"] and \
         self.all_data.loc[i, '错误方式'] != '无意义音节替换' \
         and self.all_data.loc[i, '错误方式'] != '有意义音节替换' \
         and self.all_data.loc[i, '错误方式'] != 'TEXT不在范围内'\
         and self.all_data.loc[i, '错误方式'] != '辅音缺失':
          patient_index.append(True)

      else:
        patient_index.append(False)
    subsequence_data = self.all_data[patient_index].reset_index(drop=False)
    return subsequence_data

if __name__ == '__main__':
  text_path = "data/textgrid/all_TextGrid.txt"
  TG = TextGridProcess(text_path)
  TG.read_text()
  subsequence_data = TG.delete_error()
  subsequence_data.to_csv('data/textgrid/select_TextGrid.csv', index=False, encoding='utf_8_sig')
  print("finish")