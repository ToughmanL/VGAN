import os
import csv
import numpy as np
from utils.multi_process import MultiProcess
import torch
import scipy.io.wavfile as wavfile
from spafe.features.cqcc import cqcc


class ComputeCQCC():
    def __init__(self, save_dir=None, suffix='.wav'):
        self.save_dir = save_dir
        self.wav_path_list = []
    
    def _get_files(self, data_dir, suffix):
        path_list = []
        for root, dirs, files in os.walk(data_dir, followlinks=True):
            for file in files:
                if file.endswith(suffix):
                    wav_path = os.path.join(root, file)
                    path_list.append(wav_path)
        return path_list
    
    def _cqcc_command(self, wav_path):
        if self.save_dir is not None:
            cqcc_path = os.path.join(self.save_dir, os.path.basename(wav_path).replace('.wav', '.cqcc.pt'))
        else:
            cqcc_path = wav_path.replace('.wav', '.cqcc.pt')
        if os.path.exists(cqcc_path):
            return
        sr, waveform = wavfile.read(wav_path)
        cqccs = cqcc(sig=waveform, fs=sr, num_ceps=13, pre_emph=False, nfft=1024, low_freq=50, high_freq=4000, dct_type=2, lifter=None, normalize=False, resampling_ratio=1.0,)
        tensor_cqccs = torch.tensor(cqccs)
        torch.save(tensor_cqccs, cqcc_path)
    
    def cqcc_norm(self, cqcc_dir):
        cqcc_list = self._get_files(cqcc_dir, '.cqcc.pt')
        all_data = []
        for cqcc_path in cqcc_list:
            cqcc_data = torch.load(cqcc_path)
            all_data.append(cqcc_data)
        all_data = torch.cat(all_data, dim=0)
        mean = torch.mean(all_data, dim=0)
        std = torch.std(all_data, dim=0)
        for cqcc_path in cqcc_list:
            cqcc_data = torch.load(cqcc_path)
            norm_cqcc = (cqcc_data - mean) / std
            torch.save(norm_cqcc, cqcc_path)
    
    def cqcc_count(self, wav_path_list):
        len_list = []
        for wav_path in wav_path_list:
            cqcc_path = wav_path.replace('.wav', '.cqcc.pt')
            if not os.path.exists(cqcc_path):
                print('cqcc not exist:', cqcc_path)
                continue
            name = os.path.basename(cqcc_path).split('.')[0]
            cqcc_data = torch.load(cqcc_path)
            print(name, cqcc_data.shape)
            len_list.append(cqcc_data.shape[0])
        print('min_len:', min(len_list))
        print('max_len:', max(len_list))
        print('mean_len:', np.mean(len_list))
        print('std_len:', np.std(len_list))

    
    def get_cqcc(self, multinum):
        self.wav_path_list = self._get_files(data_dir, suffix)
        if multinum == 1:
            for wav_path in self.wav_path_list:
                self._cqcc_command(wav_path)
        else:
            MP = MultiProcess()
            MP.multi_not_result(self._cqcc_command, self.wav_path_list, multinum)


if __name__ == "__main__":
    segment_dir = '/mnt/shareEEx/liuxiaokang/workspace/dysarthria-diagnosis/steps/data_process/tmp/230617_segmen_data/'
    phase_dir = '/mnt/shareEEx/liuxiaokang/workspace/av-dysarthria-diagnosis/egs/msdm/data/MSDM/crop_audio/seg_data/'
    phase_save_dir = '/mnt/shareEEx/liuxiaokang/workspace/av-dysarthria-diagnosis/egs/msdm/data/MSDM/cqcc'
    CC = ComputeCQCC()
    # CC.get_cqcc(40)
    CC.cqcc_norm(segment_dir)

    # CC = ComputeCQCC(phase_dir)
    # CC.cqcc_count(CC.wav_path_list)
    # CC.cqcc_norm(phase_save_dir)