#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	feat_dimension_reduction.py
 *@Time	: 2023-08-16 00:10:24
 *@Author	:	lxk
 *@Version	:	1.0
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''
import numpy as np

class FeatDimReduction():
  def __init__(self):
    pass
  
  def PCAFeatDimRe(self, featdata):
    from sklearn.decomposition import PCA
    six_vowel_feats = np.vstack(featdata)
    nwe_feat_data = []
    for i in range(6):
      vowel_feat = six_vowel_feats[:, i*88:(i+1)*88]
      pca = PCA(n_components=20)
      newX = pca.fit_transform(vowel_feat)
      # new_vowel_feat = pca.inverse_transform(newX)
      nwe_feat_data.append(newX)
    new_feat_arr = np.concatenate(nwe_feat_data, axis=1)
    return new_feat_arr
  
  def PCAFeatDimRe(self, featdata):
    from sklearn.decomposition import PCA
    six_vowel_feats = np.array(featdata)
    nwe_feat_data = []
    for i in range(6):
      vowel_feat = six_vowel_feats[:, i*88:(i+1)*88]
      pca = PCA(n_components=20)
      newX = pca.fit_transform(vowel_feat)
      # new_vowel_feat = pca.inverse_transform(newX)
      nwe_feat_data.append(newX)
    new_feat_arr = np.concatenate(nwe_feat_data, axis=1)
    return new_feat_arr

def test():
  featdata = []
  for i in range(1000):
    featdata.append(np.random.rand(528))
  FDR = FeatDimReduction()
  new_featdata=FDR.PCAFeatDimRe(featdata)

test()