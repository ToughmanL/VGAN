'''
提取面部坐标点
'''
import os
import cv2
import numpy as np
import pandas as pd
import json
from timeit import default_timer as timer

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from utils.multi_process import MultiProcess
from local.align_faces_Scale import align
from local.fill_null import Fill
# from show_video_coord import VideoShow
    
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
face_2d_keypoints = pipeline(Tasks.face_2d_keypoints, model='damo/cv_mobilenet_face-2d-keypoints_alignment')
class Face_feature_extract:
  def __init__(self, videos_segment_path, json_save_path, img_avi_save_path) -> None:
    self.videos_segment_path = videos_segment_path
    self.json_save_path = json_save_path
    self.img_avi_save_path = img_avi_save_path
    self.w, self.h = 800, 600
  
  #将音频的时间数据转换成视频对应的帧数,开始时间前移1帧，结束时间后移1帧
  def _time2frame(self, video_segment_time):
    time_info = video_segment_time.loc[:,["silence_end", "syllable_start", "vowel_start", "syllable_end"]]
    frame_info = (time_info * 30).astype('int')
    frame_info["syllable_end"] += 1
    return frame_info

  #获取一个syllable或者静音片段的frame list
  def _get_frame_list(self, cap, frame_start, frame_end):
    frame_list = []
    # 获取帧数
    frame_count = int(cap.get(7))
    if frame_end > frame_count:
      frame_end = frame_count
    # 指定帧
    for i_frame in range(frame_start, frame_end + 1):
      cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
      ret, frame = cap.read()  # 参数ret=True/False,代表有没有读取到图片; frame表示截取到一帧的图片
      if ret:
        frame_list.append(frame)
      else:
        frame_list.append([])
    return frame_list
  
  #检测每个syllable或静音片段的人脸坐标，返回coordinate list
  def _process_frame_list(self, frame_list):
    aligned_list = [] 
    column = ['coordinate_' + str(i) for i in range(106)]
    frame_coordinate_list = pd.DataFrame(columns=column)
    fill = Fill()
    for f in frame_list:
      #没有捕捉到帧, 对每一个坐标写入缺失值
      if len(f) == 0: 
        new = fill.fill_coord("")
        aligned_list.append([])
      else:
        try:
          #1.灰度归一化
          grayed = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
          #2.检测人脸
          output = face_2d_keypoints(grayed)
          if len(output["keypoints"]):
            frame_coordinate = np.array(np.round(output["keypoints"][0],0), dtype=int)#(1, 106, 2)
            # 1.几何归一化
            frame_align = align(grayed, frame_coordinate)
            # 2. 重新检测人脸
            output = face_2d_keypoints(frame_align)
            if len(output["keypoints"]):
              frame_coordinate = np.array(np.round(output["keypoints"][0],0), dtype=int)#(1, 106, 2)
              new = fill.fill_coord(frame_coordinate)
              aligned_list.append(frame_align)
          # 没检测到人脸
          if not len(output["keypoints"]):
            new = fill.fill_coord("")
            aligned_list.append([])
        except:
          new = fill.fill_coord("")
          aligned_list.append([])
      frame_coordinate_list = frame_coordinate_list.append(pd.DataFrame(new, index=[0]), ignore_index=True)
    return aligned_list, frame_coordinate_list
  
  #找到第一个检测点不为空的帧，填充后面的空值
  def _check_null(self, coordinate_list, aligned_list, video_name):
    null_save_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/null_face_exceed_6.txt"
    fill = Fill()
    for i, frame_coord in enumerate(coordinate_list.values):
      if not frame_coord[0] == 'nan,nan':
        aligned_list = fill.fill_null_frame(aligned_list[i:])
        coordinate_list = coordinate_list.iloc[i:,:].reset_index(drop=True)
        coordinate_list = fill.fill_null_coord(coordinate_list, video_name, null_save_path)
        break
    return aligned_list, coordinate_list  
 
  #计算言语时的segment
  def _process_segment(self, video_path, video_name, video_segment_time):
    cap = cv2.VideoCapture(video_path)
    frame_info = self._time2frame(video_segment_time)
    data_dict ={}
    frame_list = []
    coord_list = []
    #处理静止片段
    silence_frame_end = frame_info.iloc[0,0]
    silence_frame_list = self._get_frame_list(cap, 0, silence_frame_end)
    silence_aligned_list, silence_coordinate_list = self._process_frame_list(silence_frame_list)
    silence_aligned_list, silence_coordinate_list = self._check_null(silence_coordinate_list, silence_aligned_list, video_name)
    frame_list.append(silence_aligned_list)
    coord_list.append(silence_coordinate_list.values.tolist())
    silence_dict = {"silence" : silence_coordinate_list.values.tolist()}

    #处理每个segment
    segment = []
    for i, syllabel_frame in enumerate(frame_info.values):
      syllable_frame_start = syllabel_frame[1]
      vowel_frame_start = syllabel_frame[2]
      syllable_frame_end = syllabel_frame[3]

      syllable_frame_list = self._get_frame_list(cap, syllable_frame_start, syllable_frame_end)
      syllable_aligned_list, syllable_coordinate_list = self._process_frame_list(syllable_frame_list)
      syllable_aligned_list, syllable_coordinate_list = self._check_null(syllable_coordinate_list, syllable_aligned_list, video_name)
      frame_list.append(syllable_aligned_list)
      coord_list.append(syllable_coordinate_list.values.tolist())

      segment_name = video_name.split(".")[0] + "_" + video_segment_time.loc[i,"syllable"] + "_" + str(video_segment_time.loc[i,"count"])
      segment.append({"segment_name" : segment_name, "vowel_frame_start" : int(vowel_frame_start), "coordinate" : syllable_coordinate_list.values.tolist()})
    segment_dict = {"segment" : segment}
    cap.release()  
    data_dict.update(silence_dict)
    data_dict.update(segment_dict)    

    return data_dict, frame_list, coord_list

  #每个进程处理一个视频
  def _process_video(self, group):
    print("process:",os.getpid())
    group[1].sort_values(by=['count'], ascending=True, inplace= True, ignore_index = True)
    video_data = group[1]
    video_path = video_data["filepath"][0].split(".")[0] + ".avi"
    video_name = video_path.split("/")[-1]
    print("dealing:",video_name)

    #判断是否已经提取过检测点
    categ = "Control" if video_name.startswith("N") else "Patient"
    tmp = video_name.split("_")
    person = "_".join([tmp[0], tmp[2], tmp[1]])
    json_file = video_name.split(".")[0] + ".json"
    
    dest_value_path = "/".join([self.json_save_path, categ, person])
    if not os.path.exists(dest_value_path):
      os.makedirs(dest_value_path)
    dest_value_path = os.path.join(dest_value_path, json_file)
    avi_path = "/".join([self.img_avi_save_path, categ, person, video_name])
    img_path = "/".join([self.img_avi_save_path, categ, person, video_name.split(".")[0]])
    jpg_img = img_path + ".jpg"
    png_path = img_path + ".png"
    
    if not (os.path.exists(dest_value_path) and os.path.exists(avi_path) \
            and (os.path.exists(jpg_img) or os.path.exists(png_path))):
      data_dict, frame_list, coord_list = self._process_segment(video_path, video_name, video_data)
      #画出唇部特征点
      # img_show = VideoShow(self.img_avi_save_path)
      # img_show.process_sample(frame_list, coord_list, video_name)
      
      res = {"filepath" : video_path}
      res.update(data_dict)
      
      data = json.dumps(res, indent=1)
      with open(dest_value_path, 'w',newline = '\n') as f:
        f.write(data)
    print("done:", video_name)

  def multi_read_video(self, multi_num):
    data = pd.read_csv(self.videos_segment_path)
    video_group = data.groupby("filepath")

    #处理每个视频的特征是否使用多线程
    if multi_num == 1:
      for group in video_group:
        self._process_video(group)
    else:
      MP = MultiProcess()
      MP.multi_cuda_not_result(func = self._process_video, arg_list = video_group, process_num = multi_num)

if __name__ == '__main__':
    videos_segment_path = '/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/extract_syllable_time/result/230816_segment_time.csv'
    json_save_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/face_json_coord"
    img_avi_save_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/img_avi_coord"
    test = Face_feature_extract(videos_segment_path, json_save_path, img_avi_save_path)
    multi_num=1
    start = timer()
    test.multi_read_video(multi_num)
    print("time:", timer() - start)
    print('All Done')

