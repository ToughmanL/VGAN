
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
 *@File	:	checkpoint.py
 *@Time	: 2023-12-25 23:59:47
 *@Author	:	lxk
 *@Version	:	1.2
 *@Contact	:	xk.liu@siat.ac.cn
 *@License	:	(C)Copyright 2022-2025, lxk&AISML
 *@Desc: 
'''


import logging
import os
import re
import yaml
import torch
from collections import OrderedDict


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
  checkpoint = torch.load(path, map_location='cpu')
  model.load_state_dict(checkpoint, strict=False)
  info_path = re.sub('.pt$', '.yaml', path)
  configs = {}
  if os.path.exists(info_path):
    with open(info_path, 'r') as fin:
      configs = yaml.load(fin, Loader=yaml.FullLoader)
  return configs

def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
  '''
  Args:
    infos (dict or None): any info you want to save.
  '''
  logging.info('Checkpoint: save to checkpoint %s' % path)
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save(state_dict, path)
  # info_path = re.sub('.pt$', '.yaml', path)
  # if infos is None:
  #   infos = {}
  # with open(info_path, 'w') as fout:
  #   data = yaml.dump(infos)
  #   fout.write(data)

def migration(model: torch.nn.Module, path: str):
  source_dict = model.state_dict()
  target_dict = torch.load(path)
  migrat_dict = {}
  for k, v in source_dict.items():
    if k in target_dict.keys():
      if source_dict[k].shape == target_dict[k].shape:
        migrat_dict[k] = v
      else:
        print("Inconsistent parameters {}".format(k))
  model.load_state_dict(migrat_dict, strict=False)
  return model

def filter_modules(model_state_dict, modules):
  new_mods = []
  incorrect_mods = []
  mods_model = model_state_dict.keys()
  for mod in modules:
    if any(key.startswith(mod) for key in mods_model):
      new_mods += [mod]
    else:
      incorrect_mods += [mod]
  if incorrect_mods:
    logging.warning(
      "module(s) %s don't match or (partially match) "
      "available modules in model.",
      incorrect_mods,
    )
    logging.warning("for information, the existing modules in model are:")
    logging.warning("%s", mods_model)

  return new_mods

def load_trained_modules(model: torch.nn.Module, path: str, encoder_pretrain: bool):
    # Load encoder modules with pre-trained model(s).
    main_state_dict = model.state_dict()
    partial_state_dict = OrderedDict()
    key_value_match = {'ShapMatch':[], 'ShapeMismatch':[], 'KeyNotFound':[]}
    print("model(s) found for pre-initialization")
    if os.path.isfile(path):
        print('Checkpoint:  %s ' % path)
        model_state_dict = torch.load(path, map_location='cpu')
        for key, value in model_state_dict.items():
            if encoder_pretrain:
              key = 'encoder.' + key
            if key in main_state_dict:
                if value.shape == main_state_dict[key].shape:
                    key_value_match['ShapMatch'] += [key]
                    partial_state_dict[key] = value
                else:
                    key_value_match['ShapeMismatch'] += [key]
                    partial_state_dict[key] = main_state_dict[key]
            else:
                key_value_match['KeyNotFound'] += [key]
    else:
        print("model was not found : %s", path)
    
    print("%d Key(s) not found in model" % len(key_value_match['KeyNotFound']))
    print("%d Key(s) with mismatched shape" % len(key_value_match['ShapeMismatch']))
    print("%d Key(s) with matched shape" % len(key_value_match['ShapMatch']))

    model.load_state_dict(partial_state_dict, strict=False)
    configs = {}
    return configs