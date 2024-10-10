#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 crop_lip_from_video.py
* @Time 	:	 2023/07/28
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import os
import cv2
import sys
import pickle
import torch
import torchvision
import torchvision.transforms.functional as F

device = "cuda:0"


def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)


def save2pkl(filename, landmarks):
    with open(filename, 'wb') as f:
	    pickle.dump(landmarks, f)


class CropMouth():
    def __init__(self, datadir):
        self.file_list = self._read_video_files(datadir, 'crop.avi')
        pass

    def _read_video_files(self, datadir, suff='avi'):
        path_list = []
        for root, dirs, files in os.walk(datadir, followlinks=True):
            for file in files:
                phy_suff = file.split('.', 1)[1]
                if phy_suff == suff:
                    path_list.append(os.path.join(root, file))
        return path_list
    
    def _crop_lip(self, input_tensor, crop_size):
        h, w, d = input_tensor.shape
        crop_w, crop_d = crop_size
        start_w = (w - crop_w) // 2
        end_w = start_w + crop_w
        start_d = (d - crop_d) // 2
        end_d = start_d + crop_d
        cropped_tensor = input_tensor[:, start_w:end_w, start_d:end_d]
        return cropped_tensor

    def _read_and_convert_to_grayscale(self, video_path):
        cor_pt = video_path.replace('.avi', '.pt')
        if os.path.exists(cor_pt):
            return
        print(cor_pt)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert to PyTorch tensor
            tensor_frame = F.to_tensor(gray_frame)
            # Normalize the tensor (optional)
            tensor_frame = F.normalize(tensor_frame, [0.5], [0.5])
            if tensor_frame.shape[2] > 80:
                tensor_frame = self._crop_lip(tensor_frame, (80, 80))
            frames.append(tensor_frame)
        cap.release()
        if len(frames) == 0:
            print(video_path, 'read None')
            tensor_frame = torch.rand(1,80,80)
            frames.append(tensor_frame)
            frames.append(tensor_frame)
        output = torch.stack(frames)
        cor_pt = video_path.replace('.avi', '.pt')
        torch.save(output, cor_pt)

    # def _crop_save(self, avi_path):
    #     landmark_path = avi_path.replace('.avi', '.pkl')
    #     crop_avi_path = avi_path.replace('.avi', '.crop.avi')
    #     if os.path.exists(landmark_path) and os.path.exists(crop_avi_path) :
    #         return
    #     print(avi_path)
    #     landmarks_detector = LandmarksDetector(device=device)
    #     dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector='retinaface', convert_gray=False)
    #     landmarks = landmarks_detector(avi_path)
    #     data = dataloader.load_data(avi_path, landmarks)
    #     fps = cv2.VideoCapture(avi_path).get(cv2.CAP_PROP_FPS)
    #     save2vid(crop_avi_path, data, fps)
    #     save2pkl(landmark_path, landmarks)
    
    def crop_mouth_inter(self):
        for avi_path in self.file_list:
            # self._crop_save(avi_path)
            self._read_and_convert_to_grayscale(avi_path)


if __name__ == "__main__":
    # main()
    datadir = sys.argv[1]
    # datadir = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/230617_segmen_data/Patient/S_00004_M'
    print(datadir)
    CM = CropMouth(datadir)
    CM.crop_mouth_inter()
    print(datadir, 'done')
