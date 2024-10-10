import math
import numpy as np
import os
import cv2

class Calculate(object):
    # 方法类，负责特征的计算
    def __init__(self, shape):
        self.shape = shape

    def Lip_angle_minus(self, point_1, point_2, point_3):
        # 计算以point_1为顶点，point_2, point_3为底三角形两底角之差
        A = self.Lip_angle(point_2, point_1, point_3)
        B = self.Lip_angle(point_3, point_1, point_2)
        D = abs(A - B)
        # D = A / B
        return D

    def Lip_angle(self, point_1, point_2, point_3):
        # 计算以point_1为顶点，point_2, point_3为底三角形顶角
        point_1 = self.shape.part(point_1)
        point_2 = self.shape.part(point_2)
        point_3 = self.shape.part(point_3)
        a = math.sqrt(
            (point_2.x - point_3.x) * (point_2.x - point_3.x) + (point_2.y - point_3.y) * (point_2.y - point_3.y ))
        b = math.sqrt(
            (point_1.x - point_3.x) * (point_1.x - point_3.x) + (point_1.y - point_3.y) * (point_1.y - point_3.y))
        c = math.sqrt(
            (point_1.x - point_2.x) * (point_1.x - point_2.x) + (point_1.y - point_2.y) * (point_1.y - point_2.y))

        A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
        return A

    def dist(self, point_1, point_2):
        # 计算两点之间距离
        point_1 = self.shape.part(point_1)
        point_2 = self.shape.part(point_2)
        d = math.sqrt(
            (point_1.x - point_2.x) * (point_1.x - point_2.x) + (point_1.y - point_2.y) * (point_1.y - point_2.y))
        return d

    def f1(self, x, A, C):
        return A * x + C

    def fitting_line(self, points):
        # 拟合直线，返回拟合直线斜率k
        point = []
        for i in points:
            point.append([self.shape.part(i).x, self.shape.part(i).y])
        output = cv2.fitLine(np.array(point), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k * output[2]
        return k[0], b[0]

    def dist_line(self, point):
        # 计算某点到鼻子中轴线距离
        fit_point = []
        for i in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
            fit_point.append([self.shape.part(i).x, self.shape.part(i).y])
        # output:[cos a, sin a, point_x, point_y]
        output = cv2.fitLine(np.array(fit_point), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k * output[2]
        # 计算三组对称点距离之差
        dist = (math.fabs(k * self.shape.part(point).x - 1 * self.shape.part(point).y + b)) / math.sqrt(k * k + 1)

        return dist

    def dist_line_minus(self):
        # 求两点到鼻子中轴线的距离之差最大值
        # 直线拟合
        point = []
        for i in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
            point.append([self.shape.part(i).x, self.shape.part(i).y])
        # output:[cos a, sin a, point_x, point_y]
        output = cv2.fitLine(np.array(point), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k * output[2]
        # 计算三组对称点距离之差
        dist1 = abs((math.fabs(k * self.shape.part(49).x - 1 * self.shape.part(49).y + b)) / math.sqrt(k * k + 1) -
                    (math.fabs(k * self.shape.part(53).x - 1 * self.shape.part(53).y + b)) / math.sqrt(k * k + 1))
        dist2 = abs((math.fabs(k * self.shape.part(48).x - 1 * self.shape.part(48).y + b)) / math.sqrt(k * k + 1) -
                    (math.fabs(k * self.shape.part(54).x - 1 * self.shape.part(54).y + b)) / math.sqrt(k * k + 1))
        dist3 = abs((math.fabs(k * self.shape.part(59).x - 1 * self.shape.part(59).y + b)) / math.sqrt(k * k + 1) -
                    (math.fabs(k * self.shape.part(55).x - 1 * self.shape.part(55).y + b)) / math.sqrt(k * k + 1))

        value = max(dist1, dist2, dist3)
        return value

    def dist_point_line(self, point1, point2):
        # 求两点到鼻子中轴线的距离之差
        # 直线拟合
        point = []
        for i in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
            point.append([self.shape.part(i).x, self.shape.part(i).y])
        # output:[cos a, sin a, point_x, point_y]
        output = cv2.fitLine(np.array(point), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k * output[2]
        # 计算对称点到中轴线距离之差
        dist = abs((math.fabs(k * self.shape.part(point1).x - 1 * self.shape.part(point1).y + b)) / math.sqrt(k * k + 1) -
                    (math.fabs(k * self.shape.part(point2).x - 1 * self.shape.part(point2).y + b)) / math.sqrt(k * k + 1))
        return dist

    def second_max(self, m):
        # 找出数组次大值
        m1 = max(m)
        for i in range(len(m)):
            if m[i] == m1:
                m[i] = 0

        m2 = max(m)
        return m2

    def second_min(self, m):
        # 找出数组次小值
        m1 = min(m)
        for i in range(len(m)):
            if m[i] == m1:
                m[i] = max(m)

        m2 = min(m)
        return m2


    def first_order(self, matrix):
        # 求一阶矩阵
        l, c = matrix.shape
        feat = np.zeros((8, c-1))
        for i in range(c-1):
            for j in range(8):
                feat[j, i] = matrix[j, i+1] - matrix[j, i]
        return feat

    def feature_abst(self, matrix):
        # 将n*m矩阵变为n*6矩阵，6列分别为最大值、最小值、次大值、次小值、均值和方差
        feat_1 = matrix
        feat_2 = np.zeros((feat_1.shape[0], 6))
        for i in range(feat_1.shape[0]):
            feat_2[i, 0] = max(feat_1[i, :])
            feat_2[i, 1] = min(feat_1[i, :])
            feat_2[i, 2] = self.second_max(feat_1[i, :])
            feat_2[i, 3] = self.second_min(feat_1[i, :])
            feat_2[i, 4] = np.mean(feat_1[i, :])# 计算均值
            feat_2[i, 5] = np.v1ar(feat_1[i, :]) # 计算方差
            #feat_2[i, 6] = np.std(feat_1[i, :]) # 计算标准差
        return feat_2


    def max_line(self, m):
        # 求矩阵m行的最大值
        a = m.shape[0]
        feat = np.zeros((a, 1))
        for i in range(a):
            feat[i, 0] = max(m[i, :])
        return feat

    def triangle_S(self, point_1, point_2, point_3):
        # 求三角形的面积
        a = self.dist(point_1, point_2)
        b = self.dist(point_1, point_3)
        c = self.dist(point_3, point_2)

        s = (a + b + c) / 2.0
        area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
        return area
