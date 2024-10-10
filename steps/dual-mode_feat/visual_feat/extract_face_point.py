#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 extract_face_point.py
* @Time 	:	 2022/12/12
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 Extracting face feature points
'''
import dlib
import cv2
import os, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from utils.multi_process import MultiProcess
from utils.get_files_dirs import FileDir
from utils.align_faces_Scale import face_align_withmax


class ExtractFacePoint():
  '''
   @class   ExtractFacePoint
   @desc    1. 获取人脸68个特征点
            2. 填充缺失点
            3. 填充缺失帧、删除错误帧
  '''
  def __init__(self) -> None:
    self.files_list = []
    self.detector = None
    self.predictor = None
    self.point_dict = {'Filename':'nan'}
    for ii in range(0, 68):
      self.point_dict['coordinate_' + str(ii)] = 'nan,nan'
  
  def _read_files(self, datadir):
    FD = FileDir()
    FD.get_spec_files(datadir, '.avi')
    self.files_list = FD.file_path_list
  
  def _load_detector(self, predictor_path):
    self.detector = dlib.get_frontal_face_detector()  # 获得人脸特征提取器  [(x1,y1),(x2,y2)]
    self.predictor = dlib.shape_predictor(predictor_path)  # 人脸关键点检测器
  
  def _get_max_rect(self, video_path):
    a = []
    cap = cv2.VideoCapture(video_path)
    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.arange(total_num_frames):  # total_num_frames - 1
      ret, frame = cap.read()  # 参数ret=True/False,代表有没有读取到图片; frame表示截取到一帧的图片
      if ret == 0:
        continue
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      detections = self.detector(frame, 2)  # 返回人脸矩形框4点坐标, 1len(detections)为人脸个数, 1表示放大1倍再检查
      if len(detections) == 0:
        detections = self.detector(frame, 1)
        if len(detections) == 0:
          detections = self.detector(frame)
      if len(detections) == 0:
        continue

      list_rect = [detections[0].left(), detections[0].top(), detections[0].right(), detections[0].bottom()]
      a.append(list_rect)

    if len(a) == 0:
        return []
    arr = np.array(a)

    startX = np.min(arr[:, 0])
    startY = np.min(arr[:, 1])
    endX = np.max(arr[:, 2])
    endY = np.max(arr[:, 3])

    rect_max = dlib.rectangles()
    rect_max.append(dlib.rectangle(int(startX), int(startY), int(endX), int(endY)))

    return rect_max
  
  def validation(self, video_path, video_fps, align_frames, point_list):
    new_video_path = video_path.replace('.avi', '_alignpoint.avi')
    size = align_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(new_video_path, fourcc, video_fps, size)
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4 # 可以为 0 、4、8
    for frame, points in zip(align_frames, point_list):
      for key in points.keys():
        if key == 'Filename':
          continue
        point_list = points[key].split(',')
        point_list = list(map(int, point_list))
        cv2.circle(frame, (point_list[0], point_list[1]), point_size, point_color, thickness)
      img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 
      video.write(img)
    video.release()

  def _get_face_point(self, video_path):
    filename = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧速率
    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频的帧数 时长=帧数/FPS
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    failed_frame = []
    align_frames = []
    point_list = []

    if total_num_frames == 0:
      return {'point_list':point_list, 'failed_frame':failed_frame}
    rect_max = self._get_max_rect(video_path)     # 获取整个视频流最大的人脸框
    for i in np.arange(total_num_frames):      # total_num_frames - 1
      ret, frame = cap.read()  # 参数ret=True/False,代表有没有读取到图片; frame表示截取到一帧的图片
      new = self.point_dict
      new['Filename'] = filename
      if ret == 0:
        point_list.append(new)
        failed_frame.append({'Filename':filename, 'frame':str(i + 1), 'frame_num':str(total_num_frames)})
        continue
      # 彩色图片转换为灰度图片
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # 1. 检测人脸
      detections = self.detector(frame, 2)  # 返回人脸矩形框4点坐标, 1len(detections)为人脸个数, 1表示放大1倍再检查
      if len(detections) == 0:
        detections = self.detector(frame, 1)
        if len(detections) == 0:
          detections = self.detector(frame)
          if len(detections) == 0:
            detections = rect_max
      if len(detections) == 0:
        point_list.append(new)
        failed_frame.append({'Filename':filename, 'frame':str(i + 1), 'frame_num':str(total_num_frames)})
        continue

      # 2.人脸归一化
      frame_align = face_align_withmax(frame, rect_max)

      # 3. 重新检测人脸
      detections = self.detector(frame_align, 2)
      if len(detections) == 0:
        detections = self.detector(frame_align, 1)
        if len(detections) == 0:
          detections = self.detector(frame_align, 0)

      # 赋值rectangles[[(0, 0) (256, 256)]]
      if len(detections) == 0:
        detections = dlib.rectangles()
        detections.append(dlib.rectangle(0, 0, 256, 256))

      # 4. 计算特征
      if len(detections) > 0:  # len(detection)为人脸数量
        # for k, d in enumerate(detections):  # k:人脸索引 d：对应人脸4点坐标
        d = detections[0]   # 检测第一个人脸
        # Shape of the face.  返回训练好的人脸68特征点检测器：shape.parts()
        shape = self.predictor(frame_align, d)  # predictor(img, rect) rect: 人脸框的位置信息， return:68个关键点位置
        # 保存68个点坐标
        for ii in range(0, 68):
          new['coordinate_' + str(ii)] = str(shape.part(ii).x) + ',' + str(shape.part(ii).y)
        point_list.append(new)
        align_frames.append(frame_align)
      else:
        failed_frame.append({'Filename':filename, 'frame':str(i + 1), 'frame_num':str(total_num_frames)})
    self.validation(video_path, video_fps, align_frames, point_list)
    face_point = {'point_list':point_list, 'failed_frame':failed_frame}
    return face_point
  
  def _fill_missing_point(self, pd_data):
    i = 0
    continue_count_list = []
    # 从第0个找到倒数第二个
    while (i < len(pd_data) - 1):
      if pd_data.loc[i, 'coordinate_0'] != 'nan,nan':
        i = i + 1
        continue

      # 找到该缺失值之后的第一个不为缺失值的表格
      if i == 0:
        break
      # 以i为起点继续向后找连续缺失值
      continue_count = 1
      for null_index in range(i + 1, len(pd_data)):
        if pd_data.loc[null_index, 'coordinate_0'] == 'nan,nan':
          continue_count += 1
        else:
          continue_count_list.append(continue_count)
          break

      # 填充缺失值：如果只有一个缺失值，前后均值填充
      if continue_count == 1:
        for point in range(0, 68):
          x_before, y_before = pd_data.loc[i - 1, 'coordinate_' + str(point)].split(',')
          x_after, y_after = pd_data.loc[i + 1, 'coordinate_' + str(point)].split(',')
          pd_data.loc[i, 'coordinate_' + str(point)] = str(round((int(x_before) + int(x_after)) / 2)) + ',' + str(round((int(y_before) + int(y_after)) / 2))
      if continue_count > 1:
        # 如果后面全是空缺值，使用当前值填充
        if i + continue_count -1 == len(pd_data) -1 :
          for k in range(i, i + continue_count - 1):
            pd_data.loc[k] = pd_data.loc[i - 1]
        else:   # 如果后面并非全是空缺值
          # 如果连续缺失值超过一个 对i+1到i+continue_count-1的值赋予相同的值
          for k in range(i, i + continue_count - 1):
            pd_data.loc[k] = pd_data.loc[i - 1]
          # 对i+continue_count用前后均值进行填充
          for point in range(0, 68):
            x_before, y_before = pd_data.loc[i + continue_count - 2, 'coordinate_' + str(point)].split(',')
            x_after, y_after = pd_data.loc[i + continue_count, 'coordinate_' + str(point)].split(',')
            pd_data.loc[i + continue_count - 1, 'coordinate_' + str(point)] = str(round((int(x_before) + int(x_after)) / 2)) + ',' + str(round((int(y_before) + int(y_after)) / 2))
      i = i + continue_count

    # 如果最后一帧为空缺值
    if i == len(pd_data) -1:
      if pd_data.loc[i, 'coordinate_0'] == 'nan,nan':
        pd_data.loc[i] = pd_data.loc[i - 1]
    # 保存连续帧大于6的csv文件
    if continue_count_list != [] and max(continue_count_list) > 6:
      return pd.DataFrame()
    return pd_data

  def _check_face_point(self, pd_data):
    '''
      1. 填充缺失帧
      2. 填充侧脸帧
      3. 删除错误过多的数据
    '''
    return 0

  def write_face_point(self, predictor_path, datadir, result_csv, multi_num=1):
    self._read_files(datadir)
    self._load_detector(predictor_path)
    paras_list = self.files_list
    results = []
    fail_results = []
    if multi_num == 1:
      for para in paras_list:
        face_point = self._get_face_point(para)
        results += face_point['point_list']
        fail_results += face_point['failed_frame']
    else:
      MP = MultiProcess()
      rawresults = MP.multi_with_result(func=self._get_face_point, \
          arg_list=paras_list, process_num=multi_num)
      for face_point in rawresults:
        results += face_point['point_list']
        fail_results += face_point['failed_frame']

    fail_df = pd.DataFrame(fail_results)
    # fail_df.to_csv("tmp/fail_frame.csv")

    df = pd.DataFrame(results)
    # df.to_csv(result_csv)
  
  def checkfill_face_point(self, result_csv, checkfill_csv, multi_num=1):
    filenames_pd = pd.read_csv(result_csv)
    filenames = np.unique(filenames_pd['Filename'].to_numpy())
    paras_list = []
    for filename in filenames:
      file_pd = filenames_pd[filenames_pd['Filename']==filename].reset_index(drop=True)
      paras_list.append(file_pd)
    
    results_list = [pd.DataFrame()]
    if multi_num == 1:
      for para in paras_list[0:2]:
        file_pd = self._fill_missing_point(para)
        results_list.append(file_pd)
    else:
      MP = MultiProcess()
      results_list = MP.multi_with_result(func=self._fill_missing_point, arg_list=paras_list, process_num=multi_num)
    newlist = [i for i in results_list if not(i.empty)]
    new_filenames_pd = pd.concat(newlist, ignore_index=True)
    new_filenames_pd.to_csv(checkfill_csv)
  

if __name__ == "__main__":
  predictor_path = 'tmp/face_landmark/shape_predictor_68_face_landmarks.dat'
  # datadir = 'data/segment_data'
  # datadir = 'data/segment_data/Control/N_10001_F'
  # datadir = 'data/segment_data/Patient/S_00003_M/'
  datadir = 'data/segment_data/Patient/S_00015_F/'
  result_csv = 'tmp/face_points_raw.csv'
  checkfill_csv = 'tmp/face_points_check.csv'
  EFP = ExtractFacePoint()
  EFP.write_face_point(predictor_path, datadir, result_csv, 60)
  # EFP.checkfill_face_point(result_csv, checkfill_csv, 70)