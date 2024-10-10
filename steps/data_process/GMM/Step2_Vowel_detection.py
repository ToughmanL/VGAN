#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Function: 利用GMM检测浊辅音边界点


import numpy as np
import os
from six.moves import xrange
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
from multiprocessing import Process
import pandas as pd
import shutil
import warnings
warnings.filterwarnings("ignore")

def read_feat(feat_path):
    segment_feats = np.load(feat_path)
    return segment_feats.T

# def regulation(hyp_list):
#     # 找到第一个1
#     vowel_index = 0
#     while (hyp_list[vowel_index] != 1):
#         vowel_index += 1
#     for i in range(len(hyp_list)):
#         if i < vowel_index:
#             hyp_list[i] = 0
#         else:
#             hyp_list[i] = 1
#     return hyp_list, vowel_index

def regulation(hyp_list):
    mid_idx = int(len(hyp_list)/2)
    last_cons_idx = 0
    while hyp_list[mid_idx] == 0:
      mid_idx -= 1
    for i in range(mid_idx, -1, -1):
      if hyp_list[i] == 0:
        last_cons_idx = i
        break
    for i in range(len(hyp_list)):
      if i <= last_cons_idx:
        hyp_list[i] = 0
      else:
        hyp_list[i] = 1
    return hyp_list, last_cons_idx + 1


def detect_vowel(person_path, person, cate):
    wav_st = pd.DataFrame()
    for wav in os.listdir(person_path):
        # if wav == 'N_M_10010_G1_task4_1_mu4_1_Wrong0.npy':
        hyp_list = []
        feat_path = os.path.join(person_path, wav)
        feats = read_feat(feat_path)
        cons_gmm = joblib.load(os.path.join(root_path, 'data/model/gmm_NS_conso_com60_max40_covfull.smn'))
        vowel_gmm = joblib.load(os.path.join(root_path, 'data/model/gmm_NS_vowel_com70_max80_covfull.smn'))
        score1 = cons_gmm.score_samples(feats)
        score2 = vowel_gmm.score_samples(feats)
        for k in range(len(score1)):
            if score1[k] > score2[k]:   # 为什么要大于score2[1]? 而不是score2[k]
                hyp_list.append(0)
            else:
                hyp_list.append(1)
        # 保存全为0的样本
        if (np.array(hyp_list) == 0).sum() == len(hyp_list):
            print("all cons: ", wav)
        else:
            # 找到元音开始时间
            hyp, vowel_index = regulation(hyp_list)
            new = pd.DataFrame({'Filename': wav.replace('.npy', '.wav'), 'Person': person, 'start': vowel_index * 0.01}, index=[1])
            wav_st = wav_st.append(new, ignore_index=True)
    # 保存
    wav_st = wav_st.sort_values(by=['Person', 'Filename'], ascending=True).reset_index(drop=True)
    wav_st.to_csv(os.path.join(root_path, 'result/tmp/' + person + '.csv'), index=False)

def Merge(path):
    base_move_total = pd.DataFrame()
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        person_base = pd.read_csv(file_path)
        base_move_total = pd.concat([base_move_total, person_base], axis=0)
    base_move_total = base_move_total.sort_values(by=['Person', 'Filename']).reset_index(drop=True)
    return base_move_total

root_path = '/mnt/shareEEx/liuxiaokang/data/lsj_work/lxk_Lipspeech/data_process/audio_feat_process/python_script/GMM'

if __name__ == '__main__':
    feat_dir = os.path.join(root_path, 'data/gmmfeat')
    # 创建result/tmp
    res_dir = os.path.join(root_path, 'result', 'tmp')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    work = []
    for cate in os.listdir(feat_dir):
        cate_path = os.path.join(feat_dir, cate)
        for person in os.listdir(cate_path):
            # if person == 'N_10010_M':
            person_path = os.path.join(cate_path, person)
            p = Process(target=detect_vowel, args=(person_path, person, cate))
            work.append(p)
            p.start()

    for p in work:
        p.join()

    # 合并
    st_csv = Merge(os.path.join(root_path, 'result/tmp/'))
    # 删除
    shutil.rmtree(os.path.join(root_path, 'result/tmp/'))

    st_csv.to_csv(os.path.join(root_path, 'result/' + 'voiced_st_new.csv'), index=False)

    print("All done")

