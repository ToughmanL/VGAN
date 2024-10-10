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
from local.process_frame import framing, get_points
from local.Calculate_feats import Calculate


class Static:
  def __init__(self, silence_coordinate_list, wlen, inc) -> None:
     self.silence_coordinate_list = silence_coordinate_list
     self.wlen = wlen
     self.inc = inc

  #1.剔除静止片段的侧脸帧
  def _side_face_detection(self):
    side_static_frame = []
    for i, frame_coord in enumerate(self.silence_coordinate_list):
      points = get_points(frame_coord)
      # 求鼻子4个点组成的直线与竖直方向夹角
      cal = Calculate()
      k, b = cal.fitting_line((points[27], points[28], points[29], points[30]))     # 拟合直线
      # 求出中轴线与竖直方向夹角 并保存
      ver_angle = 90 - abs(np.degrees(np.arctan(k)))
      # 求出点0与点66， 点32与点79水平距离之差
      dist_x_minus = abs(abs(points[0][0] - points[36][0]) - abs(points[45][0] - points[16][0]))

      if not (ver_angle > 5.8 and dist_x_minus > 16.34):
        side_static_frame.append(i)
    # 保存剔除侧脸帧后的数据
    side_coordinate_list = [self.silence_coordinate_list[i] for i in side_static_frame]
    return side_coordinate_list, side_static_frame
  
  #计算四帧的特征
  def _get_four_feat(self, side_coordinate_list, side_static_frame, static_frame):
    static_feat = pd.DataFrame(columns=['A1_Angle_minus', 'A2_Dist_minus', 'A3_drop_dist', 'lip_width'])
    cal = Calculate()
    for sta_ind in static_frame:
      # 提取点的坐标信息
      points = get_points(side_coordinate_list[sta_ind])
      # Feature A1: 5组对称角与鼻根角度之差最大值
      A1 = max(cal.Lip_angle_minus(points[27], points[50], points[52]), cal.Lip_angle_minus(points[27], points[49], points[53]),
              cal.Lip_angle_minus(points[27], points[48], points[54]),cal.Lip_angle_minus(points[27], points[59], points[55]),
              cal.Lip_angle_minus(points[27], points[58], points[56]))
      # Feature A2: 对称点到中轴线距离之差
      A2 = cal.dist_line_minus(points)
      # Feature A3: 下垂程度(以角度和距离衡量)
      A3_dist = ((cal.dist(points[67], points[58]) + cal.dist(points[65], points[56])) / 2) / \
                ( (cal.dist(points[50], points[61]) + cal.dist(points[52], points[63])) / 2 )
      # Feature lip width
      lip_width = cal.dist(points[48], points[54])
      new = pd.DataFrame(
          {'A1_Angle_minus': A1, 'A2_Dist_minus': A2,\
          'A3_drop_dist': A3_dist, 'lip_width': lip_width}, index=[0])
      static_feat = static_feat.append(new, ignore_index=True)
    static_frame = side_static_frame[static_frame[0] : static_frame[-1]+1]
    return static_feat, static_frame

    
  #2.计算静态特征
  def _get_static_feats(self, side_coordinate_list, side_static_frame):
    cal = Calculate()
    # Step1. 计算内唇平均距离
    inner_lip_distance = []
    for frame_coord in side_coordinate_list:
      # 内唇平均距离
      points = get_points(frame_coord)
      inner_lip_distance_tmp = (cal.dist(points[61], points[67]) + cal.dist(points[62], points[66]) + cal.dist(points[63], points[65])) / 3
      inner_lip_distance.append(inner_lip_distance_tmp)

    # Step2. 选取静止状态帧并保存归一化后的图片
    inner_lip_framed, frame_index = framing(inner_lip_distance, self.wlen, self.inc)  # 对inner_lip_distance进行滑动求取标准差
    frame_std = np.std(inner_lip_framed, axis=1)    # 对每帧求标准差
    min_index = np.where(frame_std == np.min(frame_std))[0][0]  # 找到标准差最小的帧
    static_frame = frame_index[min_index]   # 将其4帧均作为静止状态
    # Step3. #计算4帧的静止状态特征A1/2/3 并保存
    static_feat, static_frame = self._get_four_feat(side_coordinate_list, side_static_frame, static_frame)
    return static_feat, static_frame

  #3.将剔除后的帧静止帧取平均
  def _four2one(self, static_feat):
    A1 = static_feat['A1_Angle_minus'].mean()
    A2 = static_feat['A2_Dist_minus'].mean()
    A3_dist = static_feat['A3_drop_dist'].mean()
    lip_width = static_feat['lip_width'].mean()
    new = pd.DataFrame({'A1_Angle_minus': A1, 'A2_Dist_minus': A2, \
                        'A3_drop_dist': A3_dist, 'lip_width': lip_width}, index=[0])
    return new


  def process_static(self):
    
    side_coordinate_list, side_static_frame = self._side_face_detection()

    static_feat, static_frame = self._get_static_feats(side_coordinate_list, side_static_frame)

    static_feat = self._four2one(static_feat)

    return static_feat, static_frame




if __name__ == '__main__':
  pass

    





    