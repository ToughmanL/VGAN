#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 average_model.py
* @Time 	:	 2023/02/27
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 average models
'''


import os
import re
import argparse
import glob
import pandas as pd

import yaml
import numpy as np
import torch


def get_args():
  parser = argparse.ArgumentParser(description='average model')
  parser.add_argument('--src_path',
            required=True,
            help='src model path for average')
  parser.add_argument('--log_path',
            required=True,
            help='log path of train')
  parser.add_argument('--val_best',
            action="store_true",
            help='averaged model')
  parser.add_argument('--num',
            default=1,
            type=int,
            help='nums for averaged model')
  parser.add_argument('--min_epoch',
            default=3,
            type=int,
            help='min epoch used for averaging model')
  parser.add_argument('--max_epoch',
            default=1000,
            type=int,
            help='max epoch used for averaging model')

  args = parser.parse_args()
  print(args)
  return args


def main():
  args = get_args()
  checkpoints = []
  val_scores = []
  df = pd.read_csv(args.log_path)
  list_dict = df.T.to_dict().values()
  for dict in list_dict:
    epoch = dict['Epoch']
    loss = dict['ValLoss']
    if epoch >= args.min_epoch and epoch <= args.max_epoch:
      val_scores += [[epoch, loss]]
  
  file_name = os.path.basename(args.log_path).split('.')[0]
  dst_model = args.log_path.replace('loss.csv', 'best.pt')
  labels = file_name.split('_')

  val_scores = np.array(val_scores)
  sort_idx = np.argsort(val_scores[:, -1])
  sorted_val_scores = val_scores[sort_idx][::1]
  print("best val scores = " + str(sorted_val_scores[:args.num, 1]))
  print("selected epochs = " + str(sorted_val_scores[:args.num, 0].astype(np.int64)))
  path_list = [
    args.src_path + '/{}_{}_{}_{}.pt'.format(labels[0], labels[1], labels[2], int(epoch))
    for epoch in sorted_val_scores[:args.num, 0]
    ]

  print(path_list)
  avg = None
  num = args.num
  assert num == len(path_list)
  for path in path_list:
    print('Processing {}'.format(path))
    states = torch.load(path, map_location=torch.device('cpu'))
    if avg is None:
      avg = states
    else:
      for k in avg.keys():
        avg[k] += states[k]
  # average
  for k in avg.keys():
    if avg[k] is not None:
      # pytorch 1.6 use true_divide instead of /=
      avg[k] = torch.true_divide(avg[k], num)
  print('Saving to {}'.format(dst_model))
  torch.save(avg, dst_model)


if __name__ == '__main__':
  main()

# python  utils/average_model.py --src_path model/dnn/ --log_path model/dnn/Frenchay_fold_0.csv