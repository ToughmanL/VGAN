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

import math
import numpy as np
import pandas as pd
import sys
from local.process_frame import get_points, outlier_detection
from local.Calculate_feats import Calculate

class Speed:
  def __init__(self) -> None:
    self.fps =30

  def _time_speak(self, arr):
    # 第一种情况: 最小值 在 最大值左侧 
    if np.argmax(arr) > np.argmin(arr):
      time = np.argmax(arr) - np.argmin(arr)
      min_frame = np.argmin(arr)
      max_frame = np.argmax(arr)

    # 第二种情况：最大值 在 最小值左侧
    else: 
      # 从最大值向左侧找
      tmp_max = arr[ : np.argmax(arr) + 1]
      k_max= np.max(tmp_max) - np.min(tmp_max)
      # 从最小值向右侧找
      tmp_min = arr[np.argmin(arr) : ]
      k_min = np.max(tmp_min) - np.min(tmp_min)
      if k_max > k_min:
        time = np.argmax(tmp_max) - np.argmin(tmp_max)
        min_frame = np.argmin(tmp_max) 
        max_frame = np.argmax(tmp_max) 
      else:
        time = np.argmax(tmp_min) - np.argmin(tmp_min)
        min_frame = np.argmin(arr) + np.argmin(tmp_min)
        max_frame = np.argmin(arr) + np.argmax(tmp_min)
        
      if time == 0 and np.argmax(arr) != np.argmin(arr):
        time = abs(np.argmax(arr) - np.argmin(arr))
        min_frame = np.argmin(arr)
        max_frame = np.argmax(arr)
    if time == 0:
      speed = None
      time = None
    else:
      speed = (arr[max_frame] - arr[min_frame]) / (time / self.fps)#张开的角度/经过的时间
    return speed, time

  def _get_speed(self, syllable_coordinate_list):
    # speed_feat = pd.DataFrame(columns=['alpha_speed', 'beta_speed', 'inner_speed', "alpha_time", "beta_time", "inner_time"])
    speed_feat = pd.DataFrame(columns=['alpha_speed', 'beta_speed', 'inner_speed'])
    cal = Calculate()
    # Step1. 提取alpha、bete、outer_lip_distance序列
    time_alpha = []
    time_beta = []
    time_inner_dist = []
    for frame_coord in syllable_coordinate_list:
      points = get_points(frame_coord)
      alpha = cal.Lip_angle(points[54], points[51], points[57])
      beta = cal.Lip_angle(points[48], points[51], points[57])
      inner_dist = (cal.dist(points[61], points[67]) + cal.dist(points[62], points[66]) + cal.dist(points[63], points[65])) / 3.0
      time_alpha.append(alpha); time_beta.append(beta)
      time_inner_dist.append(inner_dist); 

    # Step2. 异常点检测
    time_alpha_correct, _ = outlier_detection(time_alpha, k=4, rho=5)
    time_beta_correct, _ = outlier_detection(time_beta, k=4, rho=5)
    time_inner_dist_correct, _ = outlier_detection(time_inner_dist, k=4, rho=5)

    # Step3. 获取alpha、bete、outer_dis time特征
    alpha_speed, alpha_time = self._time_speak(time_alpha_correct)    # 发音阶段alpha打开时间
    beta_speed, beta_time = self._time_speak(time_beta_correct)  # 发音阶段beta打开时间
    inner_speed, inner_time = self._time_speak(time_inner_dist_correct)  # 发音阶段distance打开时间

    #Step4. 保存alpha、bete、outer_dis特征
    # new = pd.DataFrame({'alpha_speed': alpha_speed, 'beta_speed': beta_speed, 'inner_speed': inner_speed,'alpha_time': alpha_time, 'beta_time': beta_time, 'inner_time': inner_time}, index=[0])
    new = pd.DataFrame({'alpha_speed': alpha_speed, 'beta_speed': beta_speed, 'inner_speed': inner_speed}, index=[0])
    speed_feat = speed_feat.append(new, ignore_index=True)
    return speed_feat

  def process_speed(self, syllable_coordinate_list):
    speed_feat = self._get_speed(syllable_coordinate_list)
    return speed_feat

if __name__ == '__main__':

    print("All done")

