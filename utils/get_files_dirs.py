#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_files_dirs.py
@Time    :   2022/11/02 14:35:29
@Author  :   lxk 
@Version :   1.0
@Contact :   xk.liu@siat.ac.cn
@License :   (C)Copyright 2022-2025, lxk&AISML
@Desc    :   None
'''

import os

class FileDir():
  def __init__(self):
    self.files_list = []
    self.dirs_list = []
    self.file_path_list = []
    self.dir_path_list = []
  
  def _check_path(self, root_dir):
    if not os.path.exists(root_dir):
      print(root_dir, 'not exist')
      exit(-1)

  def get_dirs_files(self, root_dir):
    self._check_path(root_dir)
    for root, dirs, files in os.walk(root_dir, followlinks=True):
      for dir in dirs:
        self.dirs_list.append(dir)
        self.dir_path_list.append(os.path.join(root, dir))
      for file in files:
        self.files_list.append(file)
        self.file_path_list.append(os.path.join(root, file))
  
  def get_spec_files(self, root_dir, *suffixs):
    self._check_path(root_dir)
    for root, dirs, files in os.walk(root_dir, followlinks=True):
      for file in files:
        file_suff = file.split('.', 1)[-1]
        if file_suff in suffixs:
          self.file_path_list.append(os.path.join(root, file))
  
  def get_spec_dirs(self, root_dir, *suffixs):
    self._check_path(root_dir)
    for root, dirs, files in os.walk(root_dir, followlinks=True):
      for dir in dirs:
        for suffix in suffixs:
          if suffix in dir:
            self.dirs_list.append(dir)
            self.dir_path_list.append(os.path.join(root, dir))

