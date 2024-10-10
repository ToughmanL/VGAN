#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi_process.py
@Time    :   2022/11/03 23:40:00
@Author  :   lxk 
@Version :   1.0
@Contact :   xk.liu@siat.ac.cn
@License :   (C)Copyright 2022-2025, lxk&AISML
@Desc    :   
'''
import torch
import torch.multiprocessing as mp
from multiprocessing import Pool

class MultiProcess():
  def __init__(self):
    pass

  def multi_not_result(self, func=None, arg_list=None, process_num=30):
    if func == None or arg_list == None:
      print("func is None or arg_list is None")
      exit(-1)
    pool = Pool(processes=process_num)
    pool.map_async(func, arg_list)
    pool.close()
    pool.join()
  
  def multi_with_result(self, func=None, arg_list=None, process_num=30):
    if func == None or arg_list == None:
      print("func is None or arg_list is None")
      exit(-1)
    pool = Pool(processes=process_num)
    results = pool.map_async(func, arg_list)
    res = []
    for result in results.get():
      res.append(result)
    pool.close()
    pool.join()
    return res
  
  def multi_cuda_with_result(self, func=None, arg_list=None, process_num=4):
    if func == None or arg_list == None:
      print("func is None or arg_list is None")
      exit(-1)
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(process_num)
    results = pool.map_async(func, arg_list)
    res = []
    for result in results.get():
      res.append(result)
    pool.close()
    pool.join()
    return res

