#!/bin/bash

# nohup python hubert_wav2vec_feat.py data/result_intermediate/phase_setment_1119.csv 'cuda:0' 1>log/part_all.log 2>&1 &
# nohup python hubert_wav2vec_feat.py data/result_intermediate/phase_setment_1119.csv 'cuda:0' 1>log/part_all.log 2>&1 &

# nohup python hubert_wav2vec_feat.py data/segment_data/Control/ 'cuda:0' 1>log/part_C.log 2>&1 &
# nohup python hubert_wav2vec_feat.py data/segment_data/Patient/ 'cuda:1' 1>log/part_P.log 2>&1 &

# nohup python hubert_wav2vec_feat.py data/segment_data/Control/ 'cuda:2' 1>log/part_hubert_C.log 2>&1 &
# nohup python hubert_wav2vec_feat.py data/segment_data/Patient/ 'cuda:3' 1>log/part_hubert_P.log 2>&1 &
