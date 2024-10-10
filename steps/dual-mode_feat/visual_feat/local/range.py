#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 range.py
* @Time 	:	 2023/08/14
* @Author	:	 caodi
* @Version	:	 1.0
* @Contact	:	 caodi@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, caodi&AISML
* @Desc   	:	 None
'''


import numpy as np
import pandas as pd
import sys
from local.process_frame import get_points, framing, outlier_detection
from local.Calculate_feats import Calculate

class Range:
  def __init__(self, wlen, inc) -> None:
    self.wlen = wlen
    self.inc = inc

  #归一化
  def _range_norm(self, range_feat, lip_width):
    range_feat['w_min_norm'] = None
    range_feat['w_max_norm'] = None

    range_feat.loc[0, 'w_min_norm'] = float(range_feat.loc[0, 'w_min'] / lip_width)
    range_feat.loc[0, 'w_max_norm'] = float(range_feat.loc[0, 'w_max'] / lip_width)

    range_normalization = range_feat[['inner_dist_min', 'inner_dist_max', 'w_min_norm', 'w_max_norm']].reset_index(drop=True)
    return range_normalization
  
  #得到每个syllable的range特征
  def _get_range(self, syllable_coordinate_list):
    range_feat = pd.DataFrame(columns=['inner_dist_min', 'inner_dist_max', 'w_min', 'w_max'])
    cal = Calculate()
    # Step1. 提取inner_lip_distance序列
    w = []
    inner_lip = []
    for frame_coord in syllable_coordinate_list:
      points = get_points(frame_coord)
      inner_tmp = (cal.dist(points[61], points[67]) + cal.dist(points[62], points[66]) + cal.dist(points[63], points[65])) / 3.0
      inner_lip.append(inner_tmp)
      w_tmp = cal.dist(points[48], points[54]) # 两嘴角距离
      w.append(w_tmp)

    # Step2. 异常点检测
    inner_lip_correct, _ = outlier_detection(inner_lip, k=4, rho=5)
    w_correct, _ = outlier_detection(w, k=4, rho=5)

    # Step3. 提取innerlip 
    inner_lip_min = min(inner_lip_correct)
    inner_lip_max = max(inner_lip_correct)

    # Step4. 提取w_width 
    w_framed = framing(w_correct, self.wlen, self.inc)
    w_min = np.min(np.mean(w_framed, axis=1))
    w_max = np.max(np.mean(w_framed, axis=1))

    # Step5. 保存alpha、bete、outer_dis特征
    new = pd.DataFrame({'inner_dist_min': inner_lip_min, 'inner_dist_max': inner_lip_max, \
                        'w_min': w_min, 'w_max': w_max}, index=[0])
    range_feat = range_feat.append(new, ignore_index=True)
    return range_feat

  def process_range(self, syllable_coordinate_list, lip_width = 1):
    range_feat =self._get_range(syllable_coordinate_list)
    # range_normalization = self._range_norm(range_feat, lip_width)
    return range_feat


if __name__ == '__main__':
    print("All done")