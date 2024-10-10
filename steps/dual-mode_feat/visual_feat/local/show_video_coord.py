'''
画出每个人的一个视频中所有segment拼接后的面部坐标点视频数据+图片数据
'''
import cv2
import os
import numpy as np
from util.process_frame import get_points

class VideoShow:
  def __init__(self, save_path) -> None:
    self.save_path = save_path
    self.MARGIN = 5                # Marginal rate for image.
    self.fps = 30
    self.size = (256, 256)

  #画出面部坐标点
  def _draw_point(self, lip_keypoints, image):
    for point in lip_keypoints:
      pos=(point[0],point[1])
      cv2.circle(image,pos,1,(255,0,0),-1)
    return image

  #拼接每个segment的帧
  def _get_lip_landmark(self, frame_list, coord_list):
    # Crop images
    cropped_buffer = []
    for i, segment in enumerate(coord_list):
      tmp_frame_list = frame_list[i]
      for j, landmark in enumerate(segment):
        landmark = get_points(landmark)
        image = self._draw_point(landmark, tmp_frame_list[j])
        cropped_buffer.append(image)
    return cropped_buffer
  
  #保存为视频和图片数据
  def _write_avi_coord(self, dest_value_path, cropped_buffer):
    img_save_path = dest_value_path.split(".")[0] + ".png"
    all_frame = np.concatenate(cropped_buffer, axis=1)
    if not cv2.imwrite(img_save_path, all_frame, [cv2.IMWRITE_JPEG_QUALITY, 50]):
      raise Exception("Could not write image")
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi
    videoWriter = cv2.VideoWriter(dest_value_path, fourcc, self.fps, self.size, False) ##isColor=False 是黑白图片
    for frame in cropped_buffer:
      videoWriter.write(frame)
    videoWriter.release()
    
  #保存数据的路径
  def _save_res(self, cropped_buffer, video_name):
    categ = "Control" if video_name.startswith("N") else "Patient"
    tmp = video_name.split("_")
    person = "_".join([tmp[0], tmp[2], tmp[1]])
    avi = video_name.split(".")[0] + ".avi"
    
    dest_value_path = "/".join([self.save_path, categ, person])
    if not os.path.exists(dest_value_path):
      os.makedirs(dest_value_path)
    dest_value_path = os.path.join(dest_value_path, avi)  
    self._write_avi_coord(dest_value_path, cropped_buffer)

  def process_sample(self, frame_list, coord_list, video_name):
    cropped_buffer = self._get_lip_landmark(frame_list, coord_list)
    self._save_res(cropped_buffer, video_name)
    
if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  segment_path = "/mnt/shareEx/caodi/code/video_only/preprocess/extract_syllable_time/result/230617_gmm_con_vowel_segment.csv"
  img_save_path = "/mnt/shareEx/caodi/code/video_only/preprocess/lip_feat_extract/result/img"
  test = VideoShow(img_save_path)
  multi_num = 1