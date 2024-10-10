#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 graph_visualise.py
* @Time 	:	 2023/05/07
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''
import os
import numpy as np
import pycircos
import collections
import matplotlib.pyplot as plt
import seaborn as sns

Garc    = pycircos.Garc
Gcircle = pycircos.Gcircle

class GraphVisual():
  def __init__(self):
    self.vowels = ['a', 'o', 'e', 'i', 'u', 'v']
    self.bars = []
    self.links = []
    self.circle = Gcircle()
    self.arcdata_dict = collections.defaultdict(dict)
    self.data = None
  
  def attention_plot(self, attention, x_texts, y_texts, figsize=(15, 10), annot=False, figure_path='./figures', figure_name='attention_weight.png'):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=True,
                     fmt='.3f',
                     annot_kws={'size': 20},
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    plt.xticks(fontproperties = 'Times New Roman',size=30)
    plt.yticks(fontproperties = 'Times New Roman',size=30)
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name), bbox_inches='tight')
    plt.close()
  
  def _read_node(self, csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    head1_data = data[0::3]
    head2_data = data[1::3]
    head3_data = data[2::3]
    self.data = data
    self.bars = list(np.mean(data, axis=0))

    self.attention_plot(np.mean(head3_data, axis=0).reshape(6,6), self.vowels, self.vowels, figsize=(15, 10), annot=False, figure_path='./figures', figure_name='gat2_head3_attention_weight.png')
  
  def _read_link(self, csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    head1_data = data[0::3]
    head2_data = data[1::3]
    head3_data = data[2::3]
    self.data = data
    self.bars = list(np.mean(data, axis=0))
    
    self.attention_plot(np.mean(head1_data, axis=0).reshape(6,6), self.vowels, self.vowels, figsize=(15, 10), annot=False, figure_path='./figures', figure_name='gat2_head1_attention_weight.png')
    
    self.attention_plot(np.mean(head2_data, axis=0).reshape(6,6), self.vowels, self.vowels, figsize=(15, 10), annot=False, figure_path='./figures', figure_name='gat2_head2_attention_weight.png')
    
    self.attention_plot(np.mean(head3_data, axis=0).reshape(6,6), self.vowels, self.vowels, figsize=(15, 10), annot=False, figure_path='./figures', figure_name='gat2_head3_attention_weight.png')
  
  def barplot(self, node_path):
    self._read_node(node_path)
    color_dict = {"a":"#0FB5AE", "o":"#4046CA", "e":"#F68511", "i":"#DE3D82", "u":"#7E84FA", "v":"#72E06A"}
    for name, length in zip(self.vowels, self.bars):
      arc = Garc(arc_id=name, size=length, interspace=1, raxis_range=(500,600), labelposition=20, labelsize=15,  label_visible=True, facecolor=color_dict[name], edgecolor="#ffffff")
      self.circle.add_garc(arc)
    self.circle.set_garcs()
    self.circle.figure.savefig("tutotial_0.png")
  
  def linkplot(self, link_path):
    self._read_link(link_path)
    pass
  
  def attplot(self, gatpath):
    self._read_node(gatpath)
    for i in range(len((self.data))):
      attention = self.data[i].reshape(6,6)
      self.attention_plot(attention, self.vowels, self.vowels, figsize=(15, 10), annot=False, figure_path='./figures', figure_name='{}_attention_weight.png'.format(str(i)))

if __name__ == '__main__':
  # node_path = 'tmp/node.csv'
  gat1_path = 'tmp/gat1_line.csv'
  GV = GraphVisual()
  # GV.barplot(node_path)
  # GV.linkplot(node_path)
  # GV.attplot(gat1_path)
  # GV._read_node(node_path)
  GV._read_link(gat1_path)
