import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from util.multi_process import MultiProcess
import os
import json
import pandas as pd

from util.process_frame import get_points
from util.Calculate_feats import Calculate
# 显示中文和显示负号
# plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

class Draw:
  def __init__(self, json_path, save_path, slide_frame_count_path, new_json_path) -> None:
    self.json_path = json_path
    self.save_path = save_path
    self.slide_frame_count_path = slide_frame_count_path
    self.new_json_path = new_json_path
    self.max_angle = 5.8
    self.max_dist = 16.34

  def _get_path(self, avi_name):
    category = "Patient" if avi_name.startswith("S") else "Control"
    tmp = avi_name.split("_")
    person = "_".join([tmp[0], tmp[2], tmp[1]])
    tmp_path = os.path.join(self.save_path, category, person)
    if not os.path.exists(tmp_path):
      os.makedirs(tmp_path)
    scatter_save_path = os.path.join(tmp_path, avi_name + ".jpg")
    return scatter_save_path
  
  #画出每个视频的二维散点图
  def _plot_sactter(self, video_name, all_angle_list, all_dist_list, angle_annotate_list, dist_annotate_list, annotate_content):
    video = video_name.split(".")[0]
    
    scatter_save_path = self._get_path(video)

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize = (15, 8))
    fig.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    rcParams['font.size'] = 10

    x1 = np.arange(1, len(all_angle_list) + 1)
    x2 = np.arange(1, len(all_dist_list) + 1)
    y1 = all_angle_list
    y2 = all_dist_list
    #画出子图1
    ax1.set_title('Angle variance', fontsize = 15)
    ax1.set_xlabel('frame')
    ax1.set_ylabel('value')
    ax1.plot(x1, y1, marker='o')
    for i, xy in enumerate(angle_annotate_list):
      ax1.annotate("%s" % annotate_content[i], xy=xy, xytext=(0, 10), textcoords='offset points',\
                   bbox=dict(boxstyle='round,pad=0.1', fc='yellow', ec='k', lw=1, alpha=0.3))  
    ax1.axhline(self.max_angle, c='r', ls='--')#画出临界点的直线
    #画出子图2
    ax2.set_title('Distance variance', fontsize = 15)
    ax2.set_xlabel('frame')
    ax2.set_ylabel('value')
    ax2.plot(x2, y2, marker='o')
    for i, xy in enumerate(dist_annotate_list):
      ax2.annotate("%s" % annotate_content[i], xy=xy, xytext=(0, 10), textcoords='offset points',\
                   bbox=dict(boxstyle='round,pad=0.1', fc='yellow', ec='k', lw=1, alpha=0.3) ) 
    ax2.axhline(self.max_dist, c='r', ls='--')#画出临界点的直线

    fig.suptitle(video, fontdict={'family': 'serif', 'color': 'red', 'weight': 'bold'}, fontsize = 20)
    fig.savefig(scatter_save_path)
    plt.close()

  #计算1.人脸中轴线 2.眼角水平距离之差
  def _calculate_points(self, coordinate_list):
    angle_list = []
    dist_list = []
    slide_count = 0
    new_json_data = []
    for i, frame_coord in enumerate(coordinate_list):
      points = get_points(frame_coord)
      # 求鼻子10个点组成的直线与竖直方向夹角
      cal = Calculate()
      k, b = cal.fitting_line((points[51], points[52], points[53], points[54], points[60],\
            points[87], points[98], points[102], points[93], points[16]))     # 拟合直线
      # 求出中轴线与竖直方向夹角 并保存
      ver_angle = 90 - abs(np.degrees(np.arctan(k)))
      # 求出点0与点66， 点32与点79水平距离之差
      dist_x_minus = abs(abs(points[0][0] - points[66][0]) - abs(points[79][0] - points[32][0]))
      angle_list.append(ver_angle)
      dist_list.append(dist_x_minus)
      if ver_angle > self.max_angle and dist_x_minus > self.max_dist:
        slide_count += 1
      else:
        new_json_data.append(frame_coord) #保留剔除侧脸后的人脸数据
    return angle_list, dist_list, slide_count, new_json_data
  
  #保存新的json数据
  def _write_new_json(self, video_name, new_json_data):
    categ = "Control" if video_name.startswith("N") else "Patient"
    tmp = video_name.split("_")
    person = "_".join([tmp[0], tmp[2], tmp[1]])
    json_file = video_name.split(".")[0] + ".json"
    
    dest_value_path = "/".join([self.new_json_path, categ, person])
    if not os.path.exists(dest_value_path):
      os.makedirs(dest_value_path)
    dest_value_path = os.path.join(dest_value_path, json_file)
    data = json.dumps(new_json_data, indent=1)
    with open(dest_value_path, 'w',newline = '\n') as f:
      f.write(data)

  #从106个点中得到需要画出的人脸检测点
  def _get_draw_points(self, new_json_data, silence, segment):
    # 保存画图时annotate的数据
    angle_annotate_list = [] #保存之后需要标注的每个segment的坐标起始点
    dist_annotate_list = []
    annotate_content = []
    #计算静止状态的数据
    silence_angle_list, silence_dist_list, silence_slide_count, silence_new_json_data = self._calculate_points(silence)
    angle_annotate_list.append((1, silence_angle_list[0]))
    dist_annotate_list.append((1, silence_dist_list[0]))
    annotate_content.append("silence")
    next_annotate = len(silence_angle_list) + 1
    #保存所有坐标点
    all_angle_list = silence_angle_list
    all_dist_list = silence_dist_list
    slide_count = silence_slide_count#侧脸帧的数量
    #计算言语时的数据
    segment_data = []
    for i, syllabel_frame in enumerate(segment):
      segment_name = syllabel_frame["segment_name"].split("_")
      syllable_name = "_".join([segment_name[-2], segment_name[-1]])
      syllable_coordinate_list = syllabel_frame["coordinate"]
      syllable_angle_list, syllable_dist_list, syllable_slide_count, seg_new_json_data = self._calculate_points(syllable_coordinate_list)
      if len(seg_new_json_data)!=0:
        syllabel_frame["coordinate"] = seg_new_json_data
        segment_data.append(syllabel_frame)

      angle_annotate_list.append((next_annotate, syllable_angle_list[0]))
      dist_annotate_list.append((next_annotate, syllable_dist_list[0]))
      annotate_content.append(syllable_name)
      next_annotate += len(syllable_angle_list)

      all_angle_list += syllable_angle_list
      all_dist_list += syllable_dist_list
      slide_count += syllable_slide_count
    if len(silence_new_json_data) and len(segment_data):
      new_json_data.update({"silence" : silence_new_json_data})
      new_json_data.update({"segment" : segment_data})
    else:
      new_json_data = {}
    return all_angle_list, all_dist_list, angle_annotate_list, dist_annotate_list, annotate_content, slide_count, new_json_data

  #处理每个含人脸检测点的json文件
  def _process_avi_json(self, video_path):
    with open(video_path, 'r') as f:
      data = json.load(f)
    video_name = data["filepath"].split("/")[-1]
    silence = data["silence"]
    segment = data["segment"]
    print("dealing:",video_name)
    #得到需要的人脸检测点
    all_angle_list, all_dist_list, angle_annotate_list, dist_annotate_list, \
    annotate_content, slide_count, new_json_data = self._get_draw_points(data, silence, segment)
    #保存有侧脸帧的视频和帧数
    if len(new_json_data):
      self._write_new_json(video_name, new_json_data)
    if slide_count > 0:
      with open(self.slide_frame_count_path, 'a+') as f:
        f.write(video_name + ":" + str(slide_count) + "\n")
    # self._plot_sactter(video_name, all_angle_list, all_dist_list, angle_annotate_list, dist_annotate_list, annotate_content)
    print("done:", video_name)

  #读入所有的json文件路径
  def _read_json(self):
    json_data_path = []
    for root, dirs, files in os.walk(self.json_path):
      for file in files:
        if file.endswith(".json"):
          json_data_path.append(os.path.join(root, file))
    return json_data_path

  #多进程接口
  def multi_process(self, multi_num):
    json_data_path = self._read_json()
    #处理每个视频的检测点是否使用多线程
    if multi_num == 1:
      for video_path in json_data_path:
        self._process_avi_json(video_path)
    else:
      MP = MultiProcess()
      MP.multi_not_result(func = self._process_avi_json, arg_list = json_data_path, process_num = multi_num) 


if __name__ == '__main__':
  json_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/face_json_coord"
  save_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/scatter_fig"
  slide_frame_count_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/slide_face_count.txt"
  new_json_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/new_json_coord"
  multi_num = 1
  test =Draw(json_path, save_path, slide_frame_count_path, new_json_path)
  test.multi_process(multi_num)