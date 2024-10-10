'''
修改label可以将标签改成2或多分类
'''
import os
import pandas as pd
from sklearn import preprocessing

class Preprocess:

  def __init__(self, label, del_side_video, avi_feat_path, all_feats_save_path, \
               normal_feats_save_path, label_path, no_normal_path) -> None:
    self.avi_feat_path = avi_feat_path
    self.all_feats_save_path = all_feats_save_path
    self.normal_feats_save_path = normal_feats_save_path
    self.label_path = label_path
    self.del_side_video = del_side_video
    self.no_normal_path = no_normal_path
    self.label = label
    self.data = None
    self.Experiment_all = None

  #拼接每个视频的特征
  def get_all_feat(self):
    all_feat = pd.DataFrame(columns=["segment_name", "person", "text",\
                                      "A1_Angle_minus", "A2_Dist_minus", "A3_drop_dist", \
                                      "alpha_stability","beta_stability","dist_stability",\
                                      'alpha_speed', 'beta_speed', 'inner_speed',\
                                      "alpha_time", "beta_time", "inner_time",\
                                      "inner_dist_min","inner_dist_max","w_min_norm","w_max_norm"])
    for root, dirs, files in os.walk(self.avi_feat_path):
      for file in files:
        csv_path = os.path.join(root, file)
        one_video_feat = pd.DataFrame(pd.read_csv(csv_path))
        all_feat = all_feat.append(one_video_feat, ignore_index=True)
    all_feat.to_csv(self.all_feats_save_path, encoding="utf_8_sig", index=False) 
  
  
  
  def process_feats(self):
    self.data = pd.read_csv(self.all_feats_save_path)
    #删除侧脸帧视频
    # self.data = self._del_side_face_video()
    self.Experiment_all = self._fill_null()
    if self.label == "multi_class":
      self._add_multi_label()
    else:
      self._add_two_label()
    self.control_data = self._get_control_data()
    self._Normalization()
  
  #1.删除侧脸帧视频数据
  def _del_side_face_video(self,):
    video_list = []
    with open(self.del_side_video, 'r') as f:
      for line in f.readlines():
        video_name = line.split(".")[0]
        video_list.append(video_name)

    self.data["video_name"] = list(map(lambda x : "_".join(x.split("_")[:-2]), self.data["segment_name"]))
    aft_del_data = pd.DataFrame(columns=self.data.columns)
    for video_name in self.data["video_name"].unique():
      video_data = self.data[self.data["video_name"] == video_name].reset_index(drop=True)
      if video_name not in video_list:
        aft_del_data = pd.concat([aft_del_data, video_data], axis=0)
    aft_del_data = aft_del_data.reset_index(drop=True)
    return aft_del_data
  
  #2.填充空值：每个人数据的均值填充
  def _fill_null(self):
    # Experiment_all = self.data.dropna(axis=0, how='any')
    Experiment_all = pd.DataFrame()
    for person in self.data["person"].unique():
      person_data = self.data[self.data["person"] == person].reset_index(drop=True)
      person_data["A1_Angle_minus"].fillna(person_data["A1_Angle_minus"].mean(), inplace=True)
      person_data["A2_Dist_minus"].fillna(person_data["A2_Dist_minus"].mean(), inplace=True)
      person_data["A3_drop_dist"].fillna(person_data["A3_drop_dist"].mean(), inplace=True)
      person_data["alpha_stability"].fillna(person_data["alpha_stability"].mean(), inplace=True)
      person_data["beta_stability"].fillna(person_data["beta_stability"].mean(), inplace=True)
      person_data["dist_stability"].fillna(person_data["dist_stability"].mean(), inplace=True)
      person_data["alpha_speed"].fillna(person_data["alpha_speed"].mean(), inplace=True)
      person_data["beta_speed"].fillna(person_data["beta_speed"].mean(), inplace=True)
      person_data["inner_speed"].fillna(person_data["inner_speed"].mean(), inplace=True)
      person_data["alpha_time"].fillna(person_data["alpha_time"].mean(), inplace=True)
      person_data["beta_time"].fillna(person_data["beta_time"].mean(), inplace=True)
      person_data["inner_time"].fillna(person_data["inner_time"].mean(), inplace=True)
      person_data["inner_dist_min"].fillna(person_data["inner_dist_min"].mean(), inplace=True)
      person_data["inner_dist_max"].fillna(person_data["inner_dist_max"].mean(), inplace=True)
      person_data["w_min_norm"].fillna(person_data["w_min_norm"].mean(), inplace=True)
      person_data["w_max_norm"].fillna(person_data["w_max_norm"].mean(), inplace=True)
      Experiment_all = pd.concat([Experiment_all, person_data], axis=0)
    Experiment_all = Experiment_all.reset_index(drop=True)
    return Experiment_all

  #3. 添加标签0-3和无声调text
  def _add_multi_label(self):
    class_data = pd.read_csv(self.label_path)
    self.Experiment_all["label"] = None
    self.Experiment_all["text_no_tone"] = None

    for person in self.Experiment_all["person"].unique():
      person_class = class_data[class_data["Person"] == person].reset_index(drop=True).loc[0, "Class_number"]
      person_idx = self.Experiment_all[self.Experiment_all["person"] == person].index.tolist()
      self.Experiment_all.loc[person_idx, "label"] = person_class
      text_no_tone_list = list(map(lambda x : x[:-1], self.Experiment_all.loc[person_idx, "text"]))
      self.Experiment_all.loc[person_idx, "text_no_tone"] = text_no_tone_list

  #3. 添加标签0/1和无声调text
  def _add_two_label(self):
    class_data = pd.read_csv(self.label_path)
    self.Experiment_all["label"] = None
    self.Experiment_all["text_no_tone"] = None

    for person in self.Experiment_all["person"].unique():
      person_frenchay = class_data[class_data["Person"] == person].reset_index(drop=True).loc[0, "Frenchay"]
      # person_frenchay = class_data[class_data["Person"] == person].reset_index(drop=True).loc[0, "Class1"]
      person_idx = self.Experiment_all[self.Experiment_all["person"] == person].index.tolist()
      self.Experiment_all.loc[person_idx, "label"] = 0 if person_frenchay == 116 else 1
      # self.Experiment_all.loc[person_idx, "label"] = 0 if person_frenchay == "Co" else 1
        
      # self.Experiment_all.loc[person_idx, "label"] = 0 if person.startswith("N") else 1
      text_no_tone_list = list(map(lambda x : x[:-1], self.Experiment_all.loc[person_idx, "text"]))
      self.Experiment_all.loc[person_idx, "text_no_tone"] = text_no_tone_list
    
  #4.得到正常人的数据，为了后续的归一化
  def _get_control_data(self,):
    normal_index = []
    for i in range(len(self.Experiment_all)):
      label = self.Experiment_all.loc[i, 'label']
      index = True if label == 0 else False
      normal_index.append(index)
    control_data = self.Experiment_all[normal_index].reset_index(drop=True)
    return control_data
  
  #5. 归一化
  def _Normalization(self):
    normal_data = pd.DataFrame()
    # 按音节进行归一化
    for sylla in self.Experiment_all["text_no_tone"].unique():
      compare_data = self.Experiment_all[self.Experiment_all["text_no_tone"] == sylla].reset_index(drop=True)
      contr_data = self.control_data[self.control_data['text_no_tone'] == sylla].reset_index(drop=True)
      mms = preprocessing.StandardScaler()
      if len(contr_data) == 0:
        mms.fit(compare_data.iloc[:, 3:-2])
      else:
        mms.fit(contr_data.iloc[:, 3:-2])
      Compare_temp = pd.DataFrame(mms.transform(compare_data.iloc[:, 3:-2]),columns=compare_data.iloc[:, 3:-2].columns)

      Compare_temp = pd.concat([compare_data[["segment_name","person", "text"]], Compare_temp], axis=1)
      Compare_temp = pd.concat([Compare_temp, compare_data["label"]], axis=1)
      normal_data = pd.concat([normal_data, Compare_temp], axis=0)

    #提取出需要的视觉特征
    video_col = ["segment_name", "person", "text",\
                "A1_Angle_minus", "A2_Dist_minus", "A3_drop_dist",\
                "alpha_stability", "beta_stability", "dist_stability", \
                "alpha_time", "beta_time", "inner_time",\
                "inner_dist_min", "w_min_norm", "label"]
    # video_col = self.Experiment_all.columns[0:3].append(self.Experiment_all.columns[23:-1] )
    video_data = normal_data[video_col]
    # video_data = self.Experiment_all[video_col]
    video_data = video_data.sort_values(by=["label", "segment_name"], ascending=True).reset_index(drop=True)
    # video_data = video_data.sort_values(by=["label"], ascending=True).reset_index(drop=True)
    # video_data.to_csv(self.no_normal_path, encoding="utf_8_sig", index=False)
    video_data.to_csv(self.normal_feats_save_path, encoding="utf_8_sig", index=False)

  

if __name__ == "__main__":
  del_side_video = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/slide_face_count.txt"
  avi_feat_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/lip_feat_extract/result/all_feat_data"
  all_feats_save_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/visual_feats.csv"
  normal_feats_save_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/normal_visual_feats.csv"
  label_path = "/mnt/shareEEx/caodi/workspace/code/video_only/info/Label.csv"
  no_normal_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/no_normal_feats.csv"
  #修改此处
  label = "two_class" # "two_class"
  test = Preprocess(label, del_side_video, avi_feat_path, all_feats_save_path, normal_feats_save_path, label_path, no_normal_path)
  test.get_all_feat()
  test.process_feats()





        