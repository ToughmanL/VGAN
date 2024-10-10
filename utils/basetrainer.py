#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 trainer.py
* @Time 	:	 2023/03/26
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import torch
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class BaseTrainer():
  def __init__(self, configs, label, model, loss_fn, optimizer, scheduler):
      self.configs = configs
      self.label = label
      self.model = model
      self.loss_fn = loss_fn
      self.optimizer = optimizer
      self.scheduler = scheduler
      self.train_losses = []
      self.val_losses = []
  
  def train_step(self, x, y):
    self.model.train()
    yhat = self.model(x)
    loss = self.loss_fn(yhat, y.unsqueeze(1))
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    return loss.item()
  
  def train(self, train_loader, val_loader, device, fold_i):
    batch_size = self.configs['BATCH_SIZE']
    n_epochs = self.configs['EPOCHS']
    n_features = self.configs['NUM_FEATURES']
    train_info_list = []
    for epoch in range(1, n_epochs + 1):
      batch_losses = []
      for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        loss = self.train_step(x_batch, y_batch)
        batch_losses.append(loss)
      self.scheduler.step()
      training_loss = np.mean(batch_losses)
      self.train_losses.append(training_loss)

      with torch.no_grad():
        batch_val_losses = []
        for x_val, y_val in val_loader:
          x_val, y_val = x_val.to(device), y_val.to(device)
          self.model.eval()
          yhat = self.model(x_val)
          val_loss = self.loss_fn(yhat, y_val.unsqueeze(1)).item()
          batch_val_losses.append(val_loss)
        validation_loss = np.mean(batch_val_losses)
        self.val_losses.append(validation_loss)

      # 保存每个epoch模型
      model_path = self.configs['model_dir'] + self.label + '_fold_' + str(fold_i) + '_' + str(epoch) + '.pt'
      torch.save(self.model.state_dict(), model_path)

      train_info = f'Fold: {fold_i} | Epoch: {epoch+0:03} | Train Loss: {training_loss:4f} | Val Loss: {validation_loss:.4f}'
      print(train_info)
      train_info_list.append({'Fold':fold_i, 'Epoch':epoch, 'TrainLoss':training_loss, 'ValLoss':validation_loss})
      
    # 训练过程可视化
    train_val_loss_df = pd.DataFrame.from_dict({'train':self.train_losses, 'val':self.val_losses}).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    # plt.figure(figsize=(15,8))
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
    plt.savefig(self.configs['pic_dir'] + self.configs['model']  + '_' + self.label  + '_fold' + str(fold_i) + '.png')
    plt.clf()

    # 保存训练数据
    info_path = self.configs['model_dir'] + self.label + '_fold_' + str(fold_i) + '.csv'
    df = pd.DataFrame(train_info_list)
    df.to_csv(info_path, index=False)
  
  def evaluate(self, test_loader, device):
    batch_size = self.configs['BATCH_SIZE']
    n_features = self.configs['NUM_FEATURES']
    with torch.no_grad():
      predictions = []
      values = []
      for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        self.model.eval()
        yhat = self.model(x_test)
        predictions.append(yhat.cpu().detach().numpy())
        values.append(y_test.unsqueeze(1).cpu().detach().numpy())
    return predictions, values
  
  def plot_losses(self, pic_path):
    plt.plot(self.train_losses, label="Training loss")
    plt.plot(self.val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    # plt.show()
    plt.savefig(pic_path)
    plt.clf()