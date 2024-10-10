import math
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_visual.models.resnet import ResNet, BasicBlock
from audio_visual.models.resnet1D import ResNet1D, BasicBlock1D
from audio_visual.models.shufflenetv2 import ShuffleNetV2
from audio_visual.models.tcn import MultibranchTemporalConvNet, TemporalConvNet
from audio_visual.models.densetcn import DenseTemporalConvNet
from audio_visual.models.swish import Swish
from audio_visual.models.CustomLayer import BatchGraphAttentionLayer


# -- auxiliary functions
def threeD_to_2D_tensor(x):
  n_batch, n_channels, s_time, sx, sy = x.shape
  x = x.transpose(1, 2)
  return x.reshape(n_batch*s_time, n_channels, sx, sy)

def _average_batch(x, lengths, B):
  return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )

class MultiscaleMultibranchTCN(nn.Module):
  def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
    super(MultiscaleMultibranchTCN, self).__init__()

    self.kernel_sizes = tcn_options['kernel_size']
    self.num_kernels = len( self.kernel_sizes )

    self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
    self.tcn_output = nn.Linear(num_channels[-1], num_classes)

    self.consensus_func = _average_batch

  def forward(self, x, lengths, B):
    # x needs to have dimension (N, C, L) in order to be passed into CNN
    xtrans = x.transpose(1, 2)
    out = self.mb_ms_tcn(xtrans)
    out = self.consensus_func( out, lengths, B )
    return self.tcn_output(out)

class TCN(nn.Module):
  """Implements Temporal Convolutional Network (TCN)
  __https://arxiv.org/pdf/1803.01271.pdf
  """

  def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
    super(TCN, self).__init__()
    self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
    self.tcn_output = nn.Linear(num_channels[-1], num_classes)

    self.consensus_func = _average_batch

    self.has_aux_losses = False

  def forward(self, x, lengths, B):
    # x needs to have dimension (N, C, L) in order to be passed into CNN
    x = self.tcn_trunk(x.transpose(1, 2))
    x = self.consensus_func( x, lengths, B )
    return self.tcn_output(x)

class DenseTCN(nn.Module):
  def __init__( self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
          kernel_size_set, dilation_size_set,
          dropout, relu_type,
          squeeze_excitation=False,
    ):
    super(DenseTCN, self).__init__()

    num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
    self.tcn_trunk = DenseTemporalConvNet( block_config, growth_rate_set, input_size, reduced_size,
                      kernel_size_set, dilation_size_set,
                      dropout=dropout, relu_type=relu_type,
                      squeeze_excitation=squeeze_excitation,
                      )
    self.tcn_output = nn.Linear(num_features, num_classes)
    self.consensus_func = _average_batch

  def forward(self, x, lengths, B):
    x = self.tcn_trunk(x.transpose(1, 2))
    x = self.consensus_func( x, lengths, B )
    return self.tcn_output(x)

# 一层GAT和两层DNN结合，DNN结点64
class GAT1NN2(torch.nn.Module):
  def __init__(self, num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size):
    super().__init__()
    self.dropout = dropout
    self.attentions = [BatchGraphAttentionLayer(nfeat, nhid, dropout, batch_size, concat=True) for _ in range(nheads)] # 多头注意力
    for i, attention in enumerate(self.attentions):
      self.add_module('attention_{}'.format(i), attention)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(num_nodes*nheads*nhid, out_channels)
    self.linear2 = nn.Linear(num_nodes*nfeat, 64)
    self.linear3 = nn.Linear(64+out_channels, 32)
    self.layer_out = nn.Linear(32, 1)

  def forward(self, input):
    x = torch.cat([att(input) for att in self.attentions], dim=2) # 把每一个头计算出来的结果拼接一起
    x = F.dropout(x, 0.4, training=self.training)
    x = torch.flatten(x, 1)
    x = self.relu(self.linear1(x))

    y = torch.flatten(input, 1)
    y = self.relu(self.linear2(y))

    xy = torch.cat([x, y], dim=1)
    xy = F.dropout(xy, 0.4, training=self.training)

    xy = self.relu(self.linear3(xy))
    xy = F.dropout(xy, 0.25, training=self.training)
    # xy = self.layer_out(xy)
    return xy

class ResnetGat1nn2(nn.Module):
  def __init__( self, batch_size=32, modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500, relu_type='prelu', tcn_options={}, densetcn_options={}, width_mult=1.0, use_boundary=False, extract_feats=False):
    super(ResnetGat1nn2, self).__init__()
    self.extract_feats = extract_feats
    self.backbone_type = backbone_type
    self.modality = modality
    self.use_boundary = use_boundary

    if self.modality == 'audio':
      self.frontend_nout = 1
      self.backend_out = 512
      self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)
      # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size
      self.gat1nn2 = GAT1NN2(6, 20, 16, 32, 3, 0.4, batch_size)
    elif self.modality == 'video':
      if self.backbone_type == 'resnet':
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
      elif self.backbone_type == 'shufflenet':
        assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
        shufflenet = ShuffleNetV2( input_size=96, width_mult=width_mult)
        self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
        self.frontend_nout = 24
        self.backend_out = 1024 if width_mult != 2.0 else 2048
        self.stage_out_channels = shufflenet.stage_out_channels[-1]

      # -- frontend3D
      if relu_type == 'relu':
        frontend_relu = nn.ReLU(True)
      elif relu_type == 'prelu':
        frontend_relu = nn.PReLU( self.frontend_nout )
      elif relu_type == 'swish':
        frontend_relu = Swish()

      self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
    if self.modality == 'audio-vidio':
      if self.backbone_type == 'resnet':
        self.frontend_nout = 64
        self.backend_out = 64
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
      elif self.backbone_type == 'shufflenet':
        assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
        shufflenet = ShuffleNetV2( input_size=96, width_mult=width_mult)
        self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
        self.frontend_nout = 24
        self.backend_out = 1024 if width_mult != 2.0 else 2048
        self.stage_out_channels = shufflenet.stage_out_channels[-1]

      # -- frontend3D
      if relu_type == 'relu':
        frontend_relu = nn.ReLU(True)
      elif relu_type == 'prelu':
        frontend_relu = nn.PReLU( self.frontend_nout )
      elif relu_type == 'swish':
        frontend_relu = Swish()

      self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
      # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size
      self.gat1nn2 = GAT1NN2(6, 20, 16, 32, 3, 0.4, batch_size)

    if tcn_options:
      tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
      self.tcn = tcn_class( input_size=self.backend_out,
                  num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                  num_classes=num_classes,
                  tcn_options=tcn_options,
                  dropout=tcn_options['dropout'],
                  relu_type=relu_type,
                  dwpw=tcn_options['dwpw'],
                )
    elif densetcn_options:
      self.tcn =  DenseTCN( block_config=densetcn_options['block_config'],
                  growth_rate_set=densetcn_options['growth_rate_set'],
                  input_size=self.backend_out if not self.use_boundary else self.backend_out+1,
                  reduced_size=densetcn_options['reduced_size'],
                  num_classes=num_classes,
                  kernel_size_set=densetcn_options['kernel_size_set'],
                  dilation_size_set=densetcn_options['dilation_size_set'],
                  dropout=densetcn_options['dropout'],
                  relu_type=relu_type,
                  squeeze_excitation=densetcn_options['squeeze_excitation'],
                )
    else:
      raise NotImplementedError
    # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size
    self.adaptiveavgpool2d = torch.nn.AdaptiveAvgPool2d((8, 16))
    self.fc_v = nn.Linear(8*16*6, 16)
    self.fc_av = nn.Linear(48, 1)
    self.dropout = nn.Dropout(p=0.5)
    self.audio_out = nn.Linear(32, 1)
    self.video_out = nn.Linear(16, 1)
    # -- initialize
    self._initialize_weights_randomly()

  def forward(self, loop_feat_tensor, vowel_avi_data_tensor):
    if self.modality == 'video':
      vowel_x_list = []
      vowel_avi_data_tensor = vowel_avi_data_tensor.transpose(0,1)
      for avi_data_tensor in vowel_avi_data_tensor:
        B, C, T, H, W = avi_data_tensor.size()
        x = self.frontend3D(avi_data_tensor)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = self.adaptiveavgpool2d(x)
        x = torch.flatten(x, start_dim=1)
        vowel_x_list.append(x)
      vowel_x_tensor = torch.cat(vowel_x_list, dim=1)
      x = F.dropout(vowel_x_tensor, 0.6, training=self.training)
      x_visual = self.fc_v(x)
      x = F.dropout(x, 0.6, training=self.training)
      output = self.video_out(x_visual)
    elif self.modality == 'audio':
      x = self.gat1nn2(loop_feat_tensor)
      output = self.audio_out(x)
    elif self.modality == 'audio-vidio':
      vowel_x_list = []
      vowel_avi_data_tensor = vowel_avi_data_tensor.transpose(0,1)
      for avi_data_tensor in vowel_avi_data_tensor:
        B, C, T, H, W = avi_data_tensor.size()
        x = self.frontend3D(avi_data_tensor)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = self.adaptiveavgpool2d(x)
        x = torch.flatten(x, start_dim=1)
        vowel_x_list.append(x)
      vowel_x_tensor = torch.cat(vowel_x_list, dim=1)
      vowel_x_tensor = self.dropout(vowel_x_tensor)
      x_visual = self.fc_v(vowel_x_tensor)
      x_audio = self.gat1nn2(loop_feat_tensor)
      x_audio_visual = torch.cat((x_audio, x_visual), dim=1)
      x_audio_visual = self.dropout(x_audio_visual)
      output = self.fc_av(x_audio_visual)
    return output

  def _initialize_weights_randomly(self):
    use_sqrt = True
    if use_sqrt:
      def f(n):
        return math.sqrt( 2.0/float(n) )
    else:
      def f(n):
        return 2.0/float(n)

    for m in self.modules():
      if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        n = np.prod( m.kernel_size ) * m.out_channels
        m.weight.data.normal_(0, f(n))
        if m.bias is not None:
          m.bias.data.zero_()

      elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

      elif isinstance(m, nn.Linear):
        n = float(m.weight.data[0].nelement())
        m.weight.data = m.weight.data.normal_(0, f(n))

class TDCNNGat1nn2(nn.Module):
  def __init__( self, batch_size=32, modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500, relu_type='prelu', tcn_options={}, densetcn_options={}, width_mult=1.0, use_boundary=False, extract_feats=False):
    super(TDCNNGat1nn2, self).__init__()
    self.extract_feats = extract_feats
    self.backbone_type = backbone_type
    self.modality = modality
    self.use_boundary = use_boundary

    if self.modality == 'audio':
      self.frontend_nout = 1
      self.backend_out = 512
      # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size
      self.gat1nn2 = GAT1NN2(6, 20, 16, 32, 3, 0.4, batch_size)
    elif self.modality == 'video':
      if self.backbone_type == 'resnet':
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

      # -- frontend3D
      if relu_type == 'relu':
        frontend_relu = nn.ReLU(True)
      elif relu_type == 'prelu':
        frontend_relu = nn.PReLU( self.frontend_nout )
      elif relu_type == 'swish':
        frontend_relu = Swish()

      self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
    if self.modality == 'audio-vidio':
      if self.backbone_type == 'resnet':
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
      # -- frontend3D
      if relu_type == 'relu':
        frontend_relu = nn.ReLU(True)
      elif relu_type == 'prelu':
        frontend_relu = nn.PReLU( self.frontend_nout )
      elif relu_type == 'swish':
        frontend_relu = Swish()
      self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
      # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size
      self.gat1nn2 = GAT1NN2(6, 20, 16, 32, 3, 0.4, batch_size)

    if tcn_options:
      tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
      self.tcn = tcn_class( input_size=self.backend_out,
                  num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                  num_classes=num_classes,
                  tcn_options=tcn_options,
                  dropout=tcn_options['dropout'],
                  relu_type=relu_type,
                  dwpw=tcn_options['dwpw'],
                )
    elif densetcn_options:
      self.tcn =  DenseTCN( block_config=densetcn_options['block_config'],
                  growth_rate_set=densetcn_options['growth_rate_set'],
                  input_size=self.backend_out if not self.use_boundary else self.backend_out+1,
                  reduced_size=densetcn_options['reduced_size'],
                  num_classes=num_classes,
                  kernel_size_set=densetcn_options['kernel_size_set'],
                  dilation_size_set=densetcn_options['dilation_size_set'],
                  dropout=densetcn_options['dropout'],
                  relu_type=relu_type,
                  squeeze_excitation=densetcn_options['squeeze_excitation'],
                )
    else:
      raise NotImplementedError
    # num_nodes, nfeat, nhid, out_channels, nheads, dropout, batch_size
    self.adaptiveavgpool2d = torch.nn.AdaptiveAvgPool2d((16, 16))
    self.fc_v_share = nn.Linear(16*16, 16)
    self.fc_v = nn.Linear(6*16, 16)
    self.fc_av = nn.Linear(6*16+32, 1)
    self.dropout = nn.Dropout(p=0.5)
    self.audio_out = nn.Linear(32, 1)
    self.video_out = nn.Linear(16, 1)
    # -- initialize
    self._initialize_weights_randomly()


  def forward(self, loop_feat_tensor, vowel_avi_data_tensor):
    if self.modality == 'video':
      vowel_x_list = []
      vowel_avi_data_tensor = vowel_avi_data_tensor.transpose(0,1)
      for avi_data_tensor in vowel_avi_data_tensor:
        B, C, T, H, W = avi_data_tensor.size()
        x = self.frontend3D(avi_data_tensor)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = self.adaptiveavgpool2d(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc_v_share(x)
        vowel_x_list.append(x)
      x_visual = torch.cat(vowel_x_list, dim=1)
      x = F.dropout(x_visual, 0.6, training=self.training)
      x_visual = self.fc_v(x)
      x = F.dropout(x, 0.4, training=self.training)
      output = self.video_out(x_visual)
    elif self.modality == 'audio':
      x = self.gat1nn2(loop_feat_tensor)
      output = self.audio_out(x)
    elif self.modality == 'audio-vidio':
      vowel_x_list = []
      vowel_avi_data_tensor = vowel_avi_data_tensor.transpose(0,1)
      for avi_data_tensor in vowel_avi_data_tensor:
        B, C, T, H, W = avi_data_tensor.size()
        x = self.frontend3D(avi_data_tensor)
        Tnew = x.shape[2]  # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = self.adaptiveavgpool2d(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc_v_share(x)
        vowel_x_list.append(x)
      x_visual = torch.cat(vowel_x_list, dim=1)
      # vowel_x_tensor = self.dropout(vowel_x_tensor)
      # x_visual = self.fc_v(vowel_x_tensor)
      x_audio = self.gat1nn2(loop_feat_tensor)
      x_audio_visual = torch.cat((x_audio, x_visual), dim=1)
      x_audio_visual = self.dropout(x_audio_visual)
      output = self.fc_av(x_audio_visual)
    return output


  def _initialize_weights_randomly(self):
    use_sqrt = True
    if use_sqrt:
      def f(n):
        return math.sqrt( 2.0/float(n) )
    else:
      def f(n):
        return 2.0/float(n)

    for m in self.modules():
      if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        n = np.prod( m.kernel_size ) * m.out_channels
        m.weight.data.normal_(0, f(n))
        if m.bias is not None:
          m.bias.data.zero_()

      elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

      elif isinstance(m, nn.Linear):
        n = float(m.weight.data[0].nelement())
        m.weight.data = m.weight.data.normal_(0, f(n))

def get_model_json(json_path):
  with open(json_path, 'r') as json_file:
    args_loaded = json.load(json_file)
  if args_loaded.get('tcn_num_layers', ''):
      tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                      'kernel_size': args_loaded['tcn_kernel_size'],
                      'dropout': args_loaded['tcn_dropout'],
                      'dwpw': args_loaded['tcn_dwpw'],
                      'width_mult': args_loaded['tcn_width_mult'],
                    }
  else:
      tcn_options = {}
  if args_loaded.get('densetcn_block_config', ''):
      densetcn_options = {'block_config': args_loaded['densetcn_block_config'],
                          'growth_rate_set': args_loaded['densetcn_growth_rate_set'],
                          'reduced_size': args_loaded['densetcn_reduced_size'],
                          'kernel_size_set': args_loaded['densetcn_kernel_size_set'],
                          'dilation_size_set': args_loaded['densetcn_dilation_size_set'],
                          'squeeze_excitation': args_loaded['densetcn_se'],
                          'dropout': args_loaded['densetcn_dropout'],
                          }
  else:
      densetcn_options = {}
  return tcn_options, densetcn_options

def test_model():
  import json
  # get dctcn conf
  tcn_json = 'conf/lrw_resnet18_dctcn.json'
  tcn_options, densetcn_options = get_model_json(tcn_json)
  
  # get data
  # batch size 32
  loop_feat_tensor = torch.randn(64, 6, 20)
  avi_data_tensor = torch.randn(64, 6, 1, 120, 96, 96)

  AVmodel = ResnetGat1nn2(modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500, relu_type='prelu', tcn_options=tcn_options, densetcn_options=densetcn_options, width_mult=1.0, use_boundary=False, extract_feats=False)
  output = AVmodel.forward(loop_feat_tensor, avi_data_tensor)
  print(output)

if __name__ == "__main__":
  test_model()