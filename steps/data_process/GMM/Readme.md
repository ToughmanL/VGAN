

## 使用GMM来分类元音和辅音
   ### 数据准备
   1. 保持全部辅音和元音()
      + 包括浊辅音都予以保留
   2. 删除规则
      + 删除音节替换错误
      + 删除辅音缺失
   3. 数据量
      + 处理前 7817 条数据
      + 处理后 6906 条数据
   4. 路径
      + 代码：/mnt/shareEEx/liuxiaokang/data/lsj_work/GMM/textgridprop.py
      + 数据：/mnt/shareEEx/liuxiaokang/data/lsj_work/GMM/data/textgrid

   ### 特征提取
   1. 参数
      + 窗长 40ms, 窗移 10ms, 计算fft点数 256
   2. 特征(41维)
      + 13维mfcc, 13维mfcc一阶差分，13维二阶差分，一维pitch，一维过零率
   3. 路径
      + 代码：/mnt/shareEEx/liuxiaokang/data/lsj_work/GMM/feats.py
      + 数据：/mnt/shareEEx/liuxiaokang/data/lsj_work/GMM/data/textgrid

   ### 模型训练
   1. 输入数据
      + 病人数据和正常人数据一起训练，因为在使用中不可能知道语音来源
      + 将所有元音数据集中，所有辅音数据集中
      + 80% 片段用于训练， 20%用于测试
   2. gmm参数
      + 针对元音和辅音训练两个gmm
      + 元音gmm (224308 帧特征)
        - covariance_type=full, n_components=70, max_iter=60
        - full : 每个分量都有自己的一般协方差矩阵。
        - 70 n_components : 混合分量个数
        - 60 max_iter : EM迭代次数
      + 辅音gmm (63352 帧特征)
        - covariance_type=full, n_components=30, max_iter=20
        - full : 每个分量都有自己的一般协方差矩阵。
        - 30 n_components : 混合分量个数
        - 20 max_iter : EM迭代次数
   3. 规则
      + 结果序列中点前的最后一个辅音标签作为辅音的终止
   4. 结果和评价 
      |     规则前     | precision| recall  | f1-score |  support |
      |  ----          | ----     | ----    | ----  | ----  |
      |      0.0       | 0.51     | 0.49    |  0.50 | 16020 |
      |      1.0       | 0.86     | 0.87    |  0.86 | 55920 |
      |    accuracy    |          |         |  0.78 | 71940 |
      |   macro avg    |   0.68   | 0.68    |  0.68 | 71940 |
      | weighted avg   |   0.78   | 0.78    |  0.78 | 71940 |
      
      |     规则后     | precision| recall  | f1-score |  support |
      |  ----          | ----     | ----    | ----  | ----  |
      |      0.0       | 0.82     | 0.82    |  0.82 | 16260 |
      |      1.0       | 0.95     | 0.95    |  0.95 | 55969 |
      |    accuracy    |          |         |  0.92 | 72229 |
      |   macro avg    |   0.88   | 0.88    |  0.88 | 72229 |
      | weighted avg   |   0.92   | 0.92    |  0.92 | 72229 |
      更新代码之后准确率已经到99%了

 ### 模型使用
   1. 

