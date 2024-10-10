#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 accelerator.py
* @Time 	:	 2023/12/08
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
  def __iter__(self):
    return BackgroundGenerator(super().__iter__())


class DataPrefetcher():
  def __init__(self, loader, opt):
    self.loader = iter(loader)
    self.opt = opt
    self.stream = torch.cuda.Stream()
    # With Amp, it isn't necessary to manually convert data to half.
    # if args.fp16:
    #     self.mean = self.mean.half()
    #     self.std = self.std.half()
    self.preload()

  def preload(self):
    try:
      self.batch = next(self.loader)
    except StopIteration:
      self.batch = None
      return
    with torch.cuda.stream(self.stream):
      for k in self.batch:
        if k != 'meta':
          self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.next_input = self.next_input.half()
        # else:
        #     self.next_input = self.next_input.float()

  def next(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    batch = self.batch
    self.preload()
    return batch

'''
  usage:
  # ----改造后----
  prefetcher = DataPrefetcher(data_loader, opt)
  batch = prefetcher.next()
  iter_id = 0
  while batch is not None:
    iter_id += 1
    if iter_id >= num_iters:
      break
    run_step()
    batch = prefetcher.next()
'''
