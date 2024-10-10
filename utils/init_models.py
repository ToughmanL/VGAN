# -*- encoding: utf-8 -*-
'''
file       :init_models.py
Description:
Date       :2024/08/30 15:50:20
Author     :Toughman
version    :python3.8.9
'''


from utils.checkpoint import load_checkpoint, load_trained_modules
from local.VowelGNN import BatchGATCustom, GAT1, GAT3, GAT1NN2, LOOPDNNWAV2VEC, VGAT1NN2, GAT1NN2VHUBERT, GAT1NN2RESNET, GAT1NN2RESNETPAPI, EarlyFusion
from local.FusionNet import AVFusionNet
from local.SimpleNN import LSTM, DNN
from local.resnet import ResCNN
from local.seresnet import se_resnet
import os


def migration(model, configs, label, fold_i):
    if configs['checkpoint'] and configs['checkpoint'] is not None:
      checkpoint = configs['checkpoint'].replace('fold_0', f'fold_{fold_i}').replace('Frenchay', label)
      if os.path.exists(checkpoint):
        infos = load_checkpoint(model, checkpoint)
        print(f'Load checkpoint: {checkpoint}')
      else:
        infos = {}
        print(f'No such checkpoint: {checkpoint}')
    elif 'enc_init' in configs and configs['enc_init'] is not None:
        pretrain_model = configs['enc_init'].replace('fold_0', f'fold_{fold_i}')
        infos = load_trained_modules(model, pretrain_model, False)
        print(f'Load pretrain model: {pretrain_model}')
    else:
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    return model


def init_model(configs, name, label, fold_i):
    n_nodes = 6 if 'n_nodes' not in configs else configs['n_nodes']
    nfeat = 20 if 'nfeat' not in configs else configs['nfeat']
    if name == 'GAT':
        model = GAT(train_dataset[0].num_nodes, train_dataset[0].num_features, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'])
    elif name == 'GATCustom':
        model = BatchGATCustom(6, 20, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1':
        model = GAT1(6, 20, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1AV':
        model = GAT1(6, nfeat, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT3':
        model = GAT3(6, 20, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN2' or name == 'GAT1NN2PAPI' or name == 'GAT1NN2PHONATION' or name == 'GAT1NN2ARTICU' or name == 'GAT1NN2PROSODY' or name == 'GAT1NN2LIP' or name == 'GAT1NN2PAPILIP' or name == 'GAT1NN2HUBERT' or name == 'GAT1NN2LIPCMRLV' or name == 'GAT1NN2PAPILIPCMRLV' or name == 'GAT1NN2PAPICMRLV':
        model = GAT1NN2(n_nodes, nfeat, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN3' :
        model = GAT1NN3(6, 20, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN4':
        model = GAT1NN4(6, 20, configs['n_hidden'], configs['nn_hidden'],configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN5':
        model = GAT1NN5(6, 20, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN2MFCC' or name == 'GAT1NN2CQCC' or name == 'GAT1NN2IVECTOR':
        model = VGAT1NN2(6, configs['input_dim'], nfeat, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN2VHUBERT' or name == 'GAT1NN2CMLRV' or name == 'GAT1NN2LIPCMRLV_1':
        model = GAT1NN2VHUBERT(6, configs['input_dim'], nfeat, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    # elif name == 'GAT1NN2PAPILIPCMRLV' or name == 'GAT1NN2PAPICMRLV':
    #     model = EarlyFusion(6, configs['input_dim'], nfeat, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN2RESNET':
        model = GAT1NN2RESNET(6, configs['nfeat'], configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'GAT1NN2RESNETPAPI':
        model = GAT1NN2RESNETPAPI(6, configs['nfeat'], configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif name == 'LOOPDNNWAV2VEC':
        model = LOOPDNNWAV2VEC(6, nfeat, configs['n_hidden'], configs['out_channel'], configs['n_heads'], configs['dropout'], configs['BATCH_SIZE'])
    elif 'LSTM' in name:
        model = LSTM(nfeat, configs['n_hidden'], configs['n_layers'], configs['dropout'], configs['bidirectional'])
    elif name == 'AV_CMLRVPAPI_cat' or name == 'AV_CMLRVPAPI_cross':
        audio_conf = configs['audio']
        video_conf = configs['video']
        audio_model = GAT1NN2(n_nodes, audio_conf['nfeat'], audio_conf['n_hidden'], audio_conf['out_channel'], audio_conf['n_heads'], audio_conf['dropout'], configs['BATCH_SIZE'])
        audio_model = migration(audio_model, audio_conf, label, fold_i) # 迁移模型
        video_model = GAT1NN2VHUBERT(n_nodes, video_conf['input_dim'], video_conf['nfeat'], video_conf['n_hidden'], video_conf['out_channel'], video_conf['n_heads'], video_conf['dropout'], configs['BATCH_SIZE'])
        video_model = migration(video_model, video_conf, label, fold_i) # 迁移模型
        model = AVFusionNet(audio_model, video_model, configs['fusion_type'], audio_conf['nfeat'], audio_conf['out_channel'])
    elif name == 'AV_LIPCMLRVPAPI_cat' or name == 'AV_LIPCMLRVPAPI_add' or name == 'AV_LIPCMLRVPAPI_mul' or name == 'AV_LIPCMLRVPAPI_vaa' or name == 'AV_LIPCMLRVPAPI_vaa+a':
        audio_conf = configs['audio']
        video_conf = configs['video']
        audio_model = GAT1NN2(n_nodes, audio_conf['nfeat'], audio_conf['n_hidden'], audio_conf['out_channel'], audio_conf['n_heads'], audio_conf['dropout'], configs['BATCH_SIZE'])
        audio_model = migration(audio_model, audio_conf, label, fold_i) # 迁移模型
        video_model = GAT1NN2(n_nodes, video_conf['nfeat'], video_conf['n_hidden'], video_conf['out_channel'], video_conf['n_heads'], video_conf['dropout'], configs['BATCH_SIZE'])
        video_model = migration(video_model, video_conf, label, fold_i) # 迁移模型
        model = AVFusionNet(audio_model, video_model, configs['fusion_type'], audio_conf['nfeat'], audio_conf['out_channel'], configs['dropout'])
    elif name == 'DNN' or name == 'DNNHubert':
        model = DNN(nfeat, configs['n_hidden'], configs['dropout'])
    elif name == 'ResCNN':
        model = ResCNN()
    elif name == 'SEResNet':
        model = se_resnet()
    else:
        print('No such model')
        exit(-1)

    model = migration(model, configs, label, fold_i) # 迁移模型
    return model


def test_init_model():
    import yaml
    config_path = 'conf/train_av_papicmrlvlip_vaa.yaml'
    with open(config_path, 'r') as fin:
      configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(configs, 'AV_LIPCMLRVPAPI_vaa', 'Frenchay', 0)
    print(model)