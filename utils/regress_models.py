import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import ensemble
import joblib

class RegressModel():
  def __init__(self) -> None:
    pass
  
  def tree_regression(self):
    # 决策树回归
    from sklearn.tree import DecisionTreeRegressor
    model_DecisionTreeRegressor = DecisionTreeRegressor(criterion='squared_error', splitter='best',random_state=0)
    return model_DecisionTreeRegressor
  
  def exratree_regression(self):
    # ExtraTree极端随机树回归
    from sklearn.tree import ExtraTreeRegressor
    model_ExtraTreeRegressor = ExtraTreeRegressor(random_state=0)
    return model_ExtraTreeRegressor

  def svr_regression(self, C):
    # SVM回归
    from sklearn import svm
    model_SVR = svm.SVR(kernel='rbf', C=C)
    return model_SVR
  
  def knn_regression(self, n_neighbors):
    # KNN回归
    from sklearn import neighbors
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors) # n_neighbors default=5
    return model_KNeighborsRegressor
  
  def rf_regression(self, n_estimators):
    # 随机森林回归
    model_RandomForestRegressor = ensemble.RandomForestRegressor(criterion='mse',n_estimators=n_estimators,random_state=0) # 20个决策树
    return model_RandomForestRegressor
  
  def adaboost_regression(self, n_estimators):
    # Adaboost回归
    model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=n_estimators, loss='square',random_state=0)# 50个决策树
    return model_AdaBoostRegressor
  
  def gbdt_regression(self, n_estimators):
    # GBRT回归
    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(loss='squared_error', criterion='friedman_mse',n_estimators=n_estimators,random_state=0)# 100个决策树
    return model_GradientBoostingRegressor
  
  def bagging_regression(self):
    # Bagging回归
    model_BaggingRegressor = ensemble.BaggingRegressor(random_state=0)
    return model_BaggingRegressor
  
  def linear_regression(self):
    from sklearn.linear_model import LinearRegression
    model_linear_regression = LinearRegression()
    return model_linear_regression
  
  def save_model(self, model, model_path):
    joblib.dump(model, model_path)
  
  def load_model(self, model_path):
    model = joblib.load(model_path)
    return model

  def get_all_models(self):
    all_models = []
    # all_models.append({'name':'tree', 'model':self.tree_regression()})
    # all_models.append({'name':'exratree', 'model':self.exratree_regression()})
    # all_models.append({'name':'bagging', 'model':self.bagging_regression()})
    # all_models.append({'name':'linear', 'model':self.linear_regression()})
    # for num in range(40,140,10):
    for num in range(45,50,10):
      # all_models.append({'name':'knn_{}'.format(str(num)), 'model':self.knn_regression(num/5)})
      # all_models.append({'name':'gbdt_{}'.format(str(num)), 'model':self.gbdt_regression(num)})
      all_models.append({'name':'svr_{}'.format(str(num+10)), 'model':self.svr_regression(num)})
      # all_models.append({'name':'rf_{}'.format(str(num)), 'model':self.rf_regression(num)})
      # all_models.append({'name':'adaboost_{}'.format(str(num)), 'model':self.adaboost_regression(num)})
      
    return all_models

if __name__ == '__main__':
  RM = RegressModel()
  all_models = RM.get_all_models()
  print(all_models)