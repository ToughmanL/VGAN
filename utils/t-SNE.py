#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 t-SNE.py
* @Time 	:	 2023/12/26
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA

import pandas as pd
try:
  from utils.feat_select import score2class
except:
  from feat_select import score2class

class TSNEPLOT():
  def __init__(self):
    # 设置散点形状
    self.maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    # 设置散点颜色
    self.colors = ['green', 'black','red', 'blue']
    # 图例名称
    self.Label_Com = ['a', 'b', 'c', 'd']
    # 设置字体格式
    self.font1 = {'family': 'Times New Roman','weight': 'bold','size': 32}

  def down_sample(self, featsdf, label):
    # ALL
    featsdf.loc[(featsdf[label] >= 29) & (featsdf[label] < 58), label] = 3 # 9000
    featsdf.loc[(featsdf[label] >= 58) & (featsdf[label] < 87), label] = 2 # 14000
    featsdf.loc[(featsdf[label] >= 87) & (featsdf[label] < 116), label] = 1 # 38000
    featsdf.loc[(featsdf[label] == 116), label] = 0 # 4500

    featsdf = featsdf[featsdf[label] <= 3]
    selected_rows = featsdf.groupby(label, group_keys=False).apply(lambda x: x.sample(5000, random_state=0))
    return selected_rows

  def get_data(self, csv_path, label='Target'):
    feats_label = pd.read_csv(csv_path)
    feats_label = self.down_sample(feats_label, label)
    feat = feats_label.iloc[:, 0:-3].values
    feat = torch.tensor(feat, dtype=torch.float32)
    label = feats_label.loc[:,label].astype(int).values
    return feat, label

  def tsne(self, feat, n_com=2):
    ts = TSNE(n_components=n_com, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape)  # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

  def pca(self, feat, n_com=2):
    model = PCA(n_components=n_com)
    x_ts = model.fit_transform(feat)
    print(x_ts.shape)  # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

  def kmeans(self, data_scaled, n_clusters=4):
    cluster = KMeans(n_clusters=n_clusters,random_state=0)
    # cluster = GaussianMixture(n_components=n_clusters,random_state=0)
    predicted_labels = cluster.fit_predict(data_scaled)
    transform_data = cluster.fit_transform(data_scaled)
    return transform_data, predicted_labels

  def plotlabels(self, feat, label, title):
    True_labels = label.reshape((-1, 1))
    S_data = np.hstack((feat, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]
    for index in range(4):  # 假设总共有三个类别，类别的表示为0,1,2
      X = S_data.loc[S_data['label'] == index]['x']
      Y = S_data.loc[S_data['label'] == index]['y']
      plt.scatter(X, Y, cmap='brg', s=5, marker=self.maker[index], c=self.colors[index], edgecolors=self.colors[index], alpha=0.65)
      plt.xticks(fontsize=16)
      plt.yticks(fontsize=16)
    plt.title(title, fontdict={'family':'Times New Roman', 'size':32}, fontweight='normal', pad=20)

def plot_tsne(middle_csv, feat_type):
  TP = TSNEPLOT()
  feat, label_test = TP.get_data(middle_csv)
  fig = plt.figure(figsize=(10, 10))
  scaler = StandardScaler()# 数据标准化 StandardScaler
  feat = scaler.fit_transform(feat)
  x_final = TP.tsne(feat)
  # feat, predicted_labels = TP.kmeans(feat)
  TP.plotlabels(x_final, label_test, feat_type)
  # plt.show(fig)
  plt.savefig('middle_{}.png'.format(feat_type), dpi=300, bbox_inches='tight')
  print(middle_csv)


if __name__ == '__main__':
  middle_csv = 'models/nn_model/LOOPDNNWAV2VEC_1/middle_LOOPDNNWAV2VEC_wav2vecloop_0.5.csv'
  feat_type = 'WAV2VEC'
  plot_tsne(middle_csv, feat_type)


#  'GAT1NN2_15/middle_GAT1NN2_loop_featnorm_1.csv'
#  'GAT1NN2_GOP_1/middle_GAT1NN2_AV_gop_1.csv'
#  'GAT1NN2MFCC_1/middle_GAT1NN2MFCC_mfcc_1.csv'
#  'GAT1NN2_3/middle_GAT1NN2_egemaps_1.csv
# models/nn_model/GAT1NN2CQCC/middle_GAT1NN2CQCC_cqcc_1.csv
# models/nn_model/GAT1NN2IVECTOR/middle_GAT1NN2IVECTOR_ivector_1.csv
# models/nn_model/LOOPDNNWAV2VEC_1/middle_LOOPDNNWAV2VEC_wav2vecloop_0.5.csv
# models/nn_model/GAT1NN2HUBERT/middle_GAT1NN2HUBERT_hubert_0.5.csv
