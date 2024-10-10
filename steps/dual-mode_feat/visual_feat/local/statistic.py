'''
统计每个人的类别和segment数量
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib import axis
import json

class Statistic():
  def __init__(self, raw_feat_path, no_normal_path, statistic_fig_path, all_feats_path, all_person_count,experiment_person_path, experiment_feats_path, ttest_path):
    self.raw_feat_path = raw_feat_path
    self.no_normal_path = no_normal_path
    self.statistic_fig_path = statistic_fig_path
    self.all_feats_path = all_feats_path
    self.all_person_count = all_person_count
    self.experiment_person_path = experiment_person_path
    self.experiment_feats_path = experiment_feats_path
    self.ttest_path = ttest_path
    self.data = pd.DataFrame(pd.read_csv(self.all_feats_path))

  #统计每个人的数据条数
  def _person_syllable_count(self):
    person_group = self.data.groupby("person")
    person_name_list = []
    person_label_list = []
    segment_count_list = []
    for group in person_group:
      person_name = group[0]
      person_label = group[1].reset_index().loc[0, "label"]
      segment_count = group[1]["label"].count()
      person_name_list.append(person_name)
      person_label_list.append(person_label)
      segment_count_list.append(segment_count)
    person_data = pd.concat([pd.DataFrame(person_name_list), pd.DataFrame(person_label_list), pd.DataFrame(segment_count_list)], axis=1)
    person_data.columns =["person","label", "segment_num"]
    person_data.sort_values(by=['label','segment_num'], ascending=True, inplace= True, ignore_index = True)
    person_data.to_csv(self.all_person_count, encoding='utf_8_sig', index=False) 

  #统计25个病人和正常人的segment数据，用于后续分类实验
  def _Experiment_normal_feats(self,):
    experiment_feats = pd.DataFrame(columns=self.data.columns)
    experiment_person = pd.read_csv(self.experiment_person_path)
    experiment_seg_count = experiment_person["segment_num"].sum()
    control_seg_count = experiment_person.loc[experiment_person["label"] == 0]["segment_num"].sum()
    patient_seg_count = experiment_person.loc[experiment_person["label"] == 1]["segment_num"].sum()
    print("Experiment_all_seg_count:", experiment_seg_count)
    print("Experiment_control_seg_count:", control_seg_count)
    print("Experiment_patient_seg_count:", patient_seg_count)
    experiment_feats = experiment_feats.append(self.data[self.data["person"].isin(experiment_person["person"])])
    experiment_feats.to_csv(self.experiment_feats_path, encoding='utf_8_sig', index=False) 
  #画特征统计分布图
  def _draw_fig(self,):
    no_normal_data = pd.DataFrame(pd.read_csv(self.no_normal_path))
    feat = no_normal_data.columns[:-1]
    group = no_normal_data.groupby("label")
    control_feat = group.get_group(0)
    patient_feat = group.get_group(1)
    # patient1_feat = group.get_group("Mo")
    # patient2_feat = group.get_group("Se")
    
    for f in feat:
      control_feat_data = control_feat[f]
      patient_feat_data = patient_feat[f]
      # patient1_feat_data = patient1_feat[f]
      # patient2_feat_data = patient2_feat[f]
      # 绘制密度图
      plt.figure(figsize=(15, 10))
      sns.kdeplot(control_feat_data, fill=True)
      sns.kdeplot(patient_feat_data, fill=True)
      # sns.kdeplot(patient1_feat_data, fill=True)
      # sns.kdeplot(patient2_feat_data, fill=True)
      # 显示图形
      plt.title(f + 'Density Plot')
      plt.xlabel('Value')
      plt.ylabel('Density')
      plt.savefig(self.statistic_fig_path + f + ".jpg")
      plt.grid()
      plt.close()
  #t-test检验特征
  def _ttest_feature(self,):
    data = pd.DataFrame(pd.read_csv(self.all_feats_path))
    group = data.groupby("label")
    control = group.get_group(0)
    patient = group.get_group(1)
    feats = data.columns[3:-1]
    res = {}
    for feat in feats:
      sample1 = control[feat]
      sample2 = patient[feat]
      llevene = stats.levene(sample1,sample2)
      sample1 = np.asarray(sample1)
      sample2 = np.asarray(sample2)
      #levene检验：是否具有方差齐性            
      if llevene.__getattribute__("pvalue") > 0.05:
        r = stats.ttest_ind(sample1, sample2)
      else:
        r = stats.ttest_ind(sample1, sample2, equal_var=False)
      res.update({feat:{"levene": r, "statistic:":r.__getattribute__("statistic"), \
                  "pvalue:":r.__getattribute__("pvalue")}})
    data = json.dumps(res, indent=1)
    with open(self.ttest_path, 'w', newline = '\n') as f:
      f.write(data)

  def statistic(self):
    # self._person_syllable_count()
    self._Experiment_normal_feats()
    # self._draw_fig()
    # self._ttest_feature()

if __name__ == "__main__":
  raw_feat_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/visual_feats.csv"
  no_normal_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/no_normal_feats.csv"
  statistic_fig_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/density_fig/"
  all_feats_path="/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/normal_visual_feats.csv"
  all_person_count = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/person_seg_count.csv"
  experiment_person_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/classfi_person.csv"
  experiment_feats_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/experiment_feats.csv"
  ttest_path = "/mnt/shareEEx/caodi/workspace/code/video_only/preprocess/process_lip_feat/result/ttest_res.txt"
  
  test = Statistic(raw_feat_path, no_normal_path, statistic_fig_path, all_feats_path, all_person_count,experiment_person_path, experiment_feats_path, ttest_path)
  test.statistic()
