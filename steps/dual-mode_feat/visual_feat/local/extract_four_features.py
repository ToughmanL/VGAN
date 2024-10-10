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

'''
提取四组特征
  stability:   alpha_stability,beta_stability,dist_stability,
  speed:   alpha_speed,beta_speed,inner_speed,
  range:   inner_dist_min,inner_dist_max,w_min,w_max
'''
import os
import pandas as pd

from local.stability import Stability
from local.speed import Speed
from local.range import Range


class Four_feature_extract:
  #计算言语时的特征
  def speak_feat(self, segment, vowel_st):
    sta = Stability(6, 1)
    spe = Speed()
    ran = Range(4,2)
    #提取说话时3类特征
    vowel_frame_start = int(vowel_st * 30)
    stability_feat = sta.process_stability(segment, vowel_frame_start)
    speed_feat = spe.process_speed(segment)
    range_feat = ran.process_range(segment) 

    #拼接4类特征
    video_feat = pd.concat([stability_feat,speed_feat, range_feat],axis=1)
    res = video_feat.to_dict('records')[0]
    return res

  def test():
    pass

