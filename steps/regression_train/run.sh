#!/bin/bash

# stage
if [ $# -ne 1 ]; then
  echo "Usage: $0 <stage>"
  exit 1
fi
stage=$1
echo "stage: ${stage}"

# /mnt/shareEEx/yangyudong/liuxiaokang/regression/
if [ ${stage} -eq 1 ]; then
  # config_file=train_gat1nn2cqcc
  # config_file=train_gat1nn2ivector
  # config_file=train_lstm_cqcc
  # config_file=train_lstm_mfcc
  # config_file=train_dnn_ivector
  # config_file=train_dnn3mfcc
  # config_file=train_dnn3cqcc
  # config_file=train_rescnn_melspec
  # config_file=train_seresnet_stft
  # config_file=train_vhubert_dnn
  # config_file=train_gat1nn2vhubert
  # config_file=train_gat1nn2papi
  # config_file=train_gat1nn2phonation
  # config_file=train_gat1nn2articu
  # config_file=train_gat1nn2prosody
  # config_file=train_gat1nn2cropavi
  # config_file=train_gat1nn2cmlrv_2
  # config_file=train_gat1nn2papiavi
  # config_file=train_gat1nn2lip
  # config_file=train_gat1nn2papilip
  # config_file=train_gat1nn2papicmrlv
  # config_file=train_gat1nn2hubert
  # config_file=train_gat1nn2cmrlvlip
  # config_file=train_gat1nn2cmrlvlip_1
  # config_file=train_gat1nn2papilipcmrlv
  # config_file=train_dnnhubert
  # config_file=train_av_papicmrlvlip_add
  # config_file=train_av_papicmrlvlip_mul
  config_file=train_av_papicmrlvlip_cat
  # config_file=train_av_papicmrlvlip_vaa
  # config_file=train_av_papicmrlvlip_vaa+a
  folds=(0 1 2 3 4 5 6 7 8 9) # 0 1 2 3 4 5 6 7 8 9
  label=Frenchay
  for index in "${!folds[@]}"; do
      fold=${folds[$index]}
      gpu=$((index % 4))
      python gnn_regression.py --config conf/${config_file}.yaml --gpu ${gpu} --label ${label} --fold ${fold} >> log/${config_file}_${label}_${index}.log &
  done
  wait
  python gnn_regression.py --config conf/${config_file}.yaml --gpu 0 --label Frenchay --test_flag >> log/${config_file}.log;
fi

if [ ${stage} -eq 2 ]; then
# python gnn_regression.py --config conf/.yaml --gpu 0 --label Frenchay --test_flag --fold 0 1
config_files=(train_gat1nn2cqcc train_gat1nn2ivector train_loopwav2vec)
label=Frenchay
# for config_file in "${config_files[@]}"; do
#   python gnn_regression.py --config conf/${config_file}.yaml --gpu 3 --label ${label} --test_flag >> log/${config_file}.log &
for index in "${!config_files[@]}"; do
  config=(${config_files[$index]})
  gpu=$((index % 4))
  python gnn_regression.py --config conf/${config}.yaml --gpu ${gpu} --label ${label} --test_flag >> log/${config_file}.log &
done
wait
fi

