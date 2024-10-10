1. 计算GNE和VFER
  + local/VFER_main.m
  + local/GNE_main.m
  > matlabrun -nodesktop -nosplash -r VFER_main
  > matlabrun -nodesktop -nosplash -r GNE_main

2. 计算mfcc和egemaps
   + local/baseline_feats.py
   + mfcc和egemaps都要计算其均值和方差
   + mfcc还要统计数据，确定一个目标长度（80）

3. 计算GOP
  + 准备wav.scp text spk2utt utt2spk
  +  ~/workspace/kaldi-220606/egs/gop_speechocean762/s230307
  + 计算fbank特征, 计算GOP, 导出GOP

4. 收集临时特征(存储为json避免每一次使用都重新读取)
   + get_gvme_feats.py 

5. 计算声学基础特征
  + acoustic_base_feats.py

6. 计算二阶特征
  + acoustic_articulation_feats.py
  + acoustic_lsj_feats.py