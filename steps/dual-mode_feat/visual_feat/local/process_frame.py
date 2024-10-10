import numpy as np

# # 分帧
# def framing(x, wlen=7, inc=1):  # 定义分帧函数
#   signal_length = len(x)  # 获取信号的长度
#   if signal_length < wlen:
#     signal = x
#   else:
#     if signal_length == wlen:
#       nf = 1
#     else:
#       nf = int((signal_length - wlen) / inc + 1)  # 向下取整
#     # 对所有帧的时间点进行抽取，得到nf * wlen (帧数*帧长)长度的矩阵d, 即一行为一帧
#     d = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T  # 每帧的索引
#     # 将d转换为矩阵形式（数据类型为int类型）
#     d = np.array(d, dtype=np.int32)
#     signal = np.array(x)[d]
#   return signal, d

# 分帧
def framing(x, wlen=1, inc=1):  # 定义分帧函数
  signal_length = len(x)  # 获取信号的长度
  if signal_length <= wlen:
    nf = 1
    wlen = signal_length
    inc = 1
  else:
    nf = int((signal_length - wlen) / inc + 1)  # 向下取整
    # 对所有帧的时间点进行抽取，得到nf * wlen (帧数*帧长)长度的矩阵d, 即一行为一帧
  d = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T  # 每帧的索引
  # 将d转换为矩阵形式（数据类型为int类型）
  d = np.array(d, dtype=np.int32)
  signal = np.array(x)[d]
  return signal, d

#获取坐标元组
def get_points(data):
  Points = []
  for ii in range(0, 68):
    x, y = (int(data[ii][0]), int(data[ii][1]))
    Points.append((x, y))
  return Points
    
#异常点检测
def outlier_detection(data, k=4, rho=5):
  raw_data = data.copy()
  outlier_number = 0
  for i in range(int(k / 2), len(raw_data) - int(k / 2)):
    window_data = np.array([raw_data[i - int(k / 2)], raw_data[i - int(k / 2) + 1], raw_data[i + int(k / 2) - 1],
                            raw_data[i + int(k / 2)]])
    if abs(raw_data[i] - np.mean(window_data)) > rho * np.var(window_data):
      raw_data[i] = np.mean(window_data)
      outlier_number += 1
  return raw_data, outlier_number