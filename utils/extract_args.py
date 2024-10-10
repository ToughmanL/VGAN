#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 extract_args.py
* @Time 	:	 2023/12/08
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--label', nargs='+', default="['Frenchay', 'fanshe', 'huxi', 'chun', 'he', 'ruane', 'hou', 'she', 'yanyu']", help='label')
    parser.add_argument('--fold', nargs='+', default=['0','1','2','3','4','5','6','7','8','9'], help='fold id')
    parser.add_argument('--test_flag', action="store_true", default=False, help='fold id')
    
    args = parser.parse_args()
    return args