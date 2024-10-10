# -*- encoding: utf-8 -*-
'''
file       :move_segment_vhubert.py
Description:
Date       :2024/08/27 10:07:11
Author     :Toughman
version    :python3.8.9
'''
import os
from tqdm import tqdm

def get_all_files(dir_path):
  all_files = []
  for root, dirs, files in os.walk(dir_path):
    for file in files:
      if file.endswith('.npy'):
        all_files.append(os.path.join(root, file))
  return all_files


def get_dir_from_file(file_path):
  file_name = os.path.basename(file_path)
  name_ll = file_name.split('_')
  if len(name_ll) == 7:
    pass
  elif len(name_ll) == 8:
    name_ll = name_ll[1:]
  else:
    print(file_name, 'error file')
    return None
  N_S_name = "Control" if name_ll[0] == "N" else "Patient"
  person_name = name_ll[0] + '_' + name_ll[2] + '_' + name_ll[1]
  new_file_path = os.path.join(N_S_name, person_name, file_name.replace('.npy', '.vhubert.npy'))
  return new_file_path


def move_files(all_files, target_dir):
  for file_path in tqdm(all_files):
    new_file_path = get_dir_from_file(file_path)
    if new_file_path is None:
      continue
    target_path = os.path.join(target_dir, new_file_path)
    if os.path.exists(target_path):
      continue
    os.system('cp ' + file_path + ' ' + target_path)


if __name__ == '__main__':
  old_dir = "/mnt/shareEEx/liuxiaokang/workspace/AVSR/av_hubert/avhubert/clustering/MSDM/segment_data"
  new_dir = "data/segment_data/"
  all_files = get_all_files(old_dir)
  move_files(all_files, new_dir)