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
from local.process_frame import framing, get_points, outlier_detection
from local.Calculate_feats import Calculate

class Stability:
  def __init__(self, wlen, inc) -> None:
    self.wlen = wlen
    self.inc = inc


  def _process_vowel_stability(self, stability_alpha_correct, stability_beta_correct, stability_dist_correct, vowel_data, vowel_frame_start):
    spk_franum = len(vowel_data)
    spk_mid = round(spk_franum / 2)
    start_index = spk_mid - round(spk_franum * 0.8 / 2) + vowel_frame_start #在整个片段中的index
    end_index = spk_mid + round(spk_franum * 0.8 / 2) + vowel_frame_start
    # 如果spk的80% < 2
    if end_index - start_index < 2:
      if spk_franum < 2:
        alpha_stability = None
        beta_stability = None
        dist_stability = None
      else:
        alpha_stability = np.var(stability_alpha_correct)
        beta_stability = np.var(stability_beta_correct)
        dist_stability = np.var(stability_dist_correct)
    # 如果spk的80% >= 2 并且 <= 6
    elif end_index - start_index >= 2 and end_index - start_index <= 6:
      alpha_stability = np.var(stability_alpha_correct[start_index : end_index])
      beta_stability = np.var(stability_beta_correct[start_index : end_index])
      dist_stability = np.var(stability_dist_correct[start_index : end_index])
    # 如果spk的80% > 6
    else:
      # 发音阶段alpha稳定性
      alpha_framed, _ = framing(stability_alpha_correct[start_index : end_index], 6, 1)
      alpha_stability = np.mean(np.var(alpha_framed, axis=1))
      # 发音阶段beta稳定性
      beta_framed, _ = framing(stability_beta_correct[start_index : end_index], 6, 1)
      beta_stability = np.mean(np.var(beta_framed, axis=1))
      # 发音阶段distance稳定性
      outer_lip_framed, _ = framing(stability_dist_correct[start_index : end_index], 6, 1)
      dist_stability = np.mean(np.var(outer_lip_framed, axis=1))
    return alpha_stability, beta_stability, dist_stability 

  def _get_stability(self, syllable_coordinate_list, vowel_frame_start):
    stability_feat = pd.DataFrame(columns=['alpha_stability', 'beta_stability', 'dist_stability'])
    cal = Calculate()
    # Step1. 提取alpha、bete、outer_lip_distance序列
    stability_alpha = []
    stability_beta = []
    stability_dist = []
    
    for frame_coord in syllable_coordinate_list:
      points = get_points(frame_coord)
      alpha = cal.Lip_angle(points[54], points[51], points[57])
      beta = cal.Lip_angle(points[48], points[51], points[57])
      inner_dist = (cal.dist(points[61], points[67]) + cal.dist(points[62], points[66]) + cal.dist(points[63], points[65])) / 3.0
      stability_alpha.append(alpha); stability_beta.append(beta); stability_dist.append(inner_dist)

    # Step2. 异常点检测
    stability_alpha_correct, _ = outlier_detection(stability_alpha, k=4, rho=5)
    stability_beta_correct, _ = outlier_detection(stability_beta, k=4, rho=5)
    stability_dist_correct, _ = outlier_detection(stability_dist, k=4, rho=5)

    # Step3. 分情况获取alpha、bete、inner_dis stability特征
    #获取元音发音时间
    vowel_data = syllable_coordinate_list[vowel_frame_start:]
    alpha_stability, beta_stability, dist_stability = self._process_vowel_stability(\
      stability_alpha_correct, stability_beta_correct, stability_dist_correct, vowel_data, vowel_frame_start)

    #Step4. 保存alpha、bete、outer_dis特征
    new = pd.DataFrame({'alpha_stability': alpha_stability, 'beta_stability': beta_stability,
                        'dist_stability': dist_stability}, index=[0])
    stability_feat = stability_feat.append(new, ignore_index=True)
    return stability_feat


  def process_stability(self, syllable_coordinate_list, vowel_frame_start):
    stability_feat = self._get_stability(syllable_coordinate_list, vowel_frame_start)
    stability_feat.loc[stability_feat['alpha_stability'] == 0, 'alpha_stability'] = None
    stability_feat.loc[stability_feat['beta_stability'] == 0, 'beta_stability'] = None
    stability_feat.loc[stability_feat['dist_stability'] == 0, 'dist_stability'] = None
    return stability_feat

  

if __name__ == '__main__':
  
  print("All done")
