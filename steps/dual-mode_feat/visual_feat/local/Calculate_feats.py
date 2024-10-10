import math
import numpy as np
import os
import cv2

class Calculate():

    def Lip_angle_minus(self, point_1, point_2, point_3):
        # 计算以point_1为顶点，point_2, point_3为底三角形两底角之差
        A = self.Lip_angle(point_2, point_1, point_3)
        B = self.Lip_angle(point_3, point_1, point_2)
        D = abs(A - B)
        # D = A / B
        return D

    def Lip_angle(self, point_1, point_2, point_3):
        # 计算以point_1为顶点，point_2, point_3为底三角形顶角

        a = math.sqrt(
            (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
        b = math.sqrt(
            (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
        c = math.sqrt(
            (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))

        cos_val = (a * a - b * b - c * c) / (-2 * b * c)
        if cos_val > 1:
            return None
        elif cos_val == 1:
            return 0.00001
        else:
            A = math.degrees(math.acos(round(cos_val, 3)))
        return A

    def dist(self, point_1, point_2):
        # 计算两点之间距离
        d = math.sqrt(
            (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
        return d

    def f1(self, x, A, C):
        return A * x + C

    def fitting_line(self, points):
        # 拟合直线，返回拟合直线斜率k
        point = []
        for tmp_point in points:
            point.append([tmp_point[0], tmp_point[1]])
        output = cv2.fitLine(np.array(point), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k * output[2]
        return k[0], b[0]

    # def dist_line(self, point):
    #     # 计算某点到鼻子中轴线距离
    #     fit_point = []
    #     for i in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
    #         fit_point.append([self.shape.part(i).x, self.shape.part(i).y])
    #     # output:[cos a, sin a, point_x, point_y]
    #     output = cv2.fitLine(np.array(fit_point), cv2.DIST_L2, 0, 0.01, 0.01)
    #     k = output[1] / output[0]
    #     b = output[3] - k * output[2]
    #     # 计算三组对称点距离之差
    #     dist = (math.fabs(k * self.shape.part(point).x - 1 * self.shape.part(point).y + b)) / math.sqrt(k * k + 1)
    #
    #     return dist

    def dist_line_minus(self, points):
        # 求两点到鼻子中轴线的距离之差最大值
        # 直线拟合
        point = []
        for i in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
            point.append([points[i][0], points[i][1]])
        # output:[cos a, sin a, point_x, point_y]
        output = cv2.fitLine(np.array(point), cv2.DIST_L2, 0, 0.01, 0.01)
        k = output[1] / output[0]
        b = output[3] - k * output[2]
        # 计算三组对称点距离之差
        dist1 = abs((math.fabs(k * points[49][0] - 1 * points[49][1] + b)) / math.sqrt(k * k + 1) -
                    (math.fabs(k * points[53][0] - 1 * points[53][1] + b)) / math.sqrt(k * k + 1))
        dist2 = abs((math.fabs(k * points[48][0] - 1 * points[48][1] + b)) / math.sqrt(k * k + 1) -
                    (math.fabs(k * points[54][0] - 1 * points[54][1] + b)) / math.sqrt(k * k + 1))
        dist3 = abs((math.fabs(k * points[59][0] - 1 * points[59][1] + b)) / math.sqrt(k * k + 1) -
                    (math.fabs(k * points[55][0] - 1 * points[55][1] + b)) / math.sqrt(k * k + 1))

        value = max(dist1, dist2, dist3)
        return value

    # def dist_point_line(self, point1, point2):
    #     # 求两点到鼻子中轴线的距离之差
    #     # 直线拟合
    #     point = []
    #     for i in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
    #         point.append([self.shape.part(i).x, self.shape.part(i).y])
    #     # output:[cos a, sin a, point_x, point_y]
    #     output = cv2.fitLine(np.array(point), cv2.DIST_L2, 0, 0.01, 0.01)
    #     k = output[1] / output[0]
    #     b = output[3] - k * output[2]
    #     # 计算对称点到中轴线距离之差
    #     dist = abs((math.fabs(k * self.shape.part(point1).x - 1 * self.shape.part(point1).y + b)) / math.sqrt(k * k + 1) -
    #                 (math.fabs(k * self.shape.part(point2).x - 1 * self.shape.part(point2).y + b)) / math.sqrt(k * k + 1))
    #     return dist

    def first_order(self, matrix):
        # 求一阶矩阵
        l, c = matrix.shape
        feat = np.zeros((8, c-1))
        for i in range(c-1):
            for j in range(8):
                feat[j, i] = matrix[j, i+1] - matrix[j, i]
        return feat

    # def feature_abst(self, matrix):
    #     # 将n*m矩阵变为n*6矩阵，6列分别为最大值、最小值、次大值、次小值、均值和方差
    #     feat_1 = matrix
    #     feat_2 = np.zeros((feat_1.shape[0], 6))
    #     for i in range(feat_1.shape[0]):
    #         feat_2[i, 0] = max(feat_1[i, :])
    #         feat_2[i, 1] = min(feat_1[i, :])
    #         feat_2[i, 2] = self.second_max(feat_1[i, :])
    #         feat_2[i, 3] = self.second_min(feat_1[i, :])
    #         feat_2[i, 4] = np.mean(feat_1[i, :])# 计算均值
    #         feat_2[i, 5] = np.v1ar(feat_1[i, :]) # 计算方差
    #         #feat_2[i, 6] = np.std(feat_1[i, :]) # 计算标准差
    #     return feat_2


    def max_line(self, m):
        # 求矩阵m行的最大值
        a = m.shape[0]
        feat = np.zeros((a, 1))
        for i in range(a):
            feat[i, 0] = max(m[i, :])
        return feat

    # def triangle_S(self, point_1, point_2, point_3):
    #     # 求三角形的面积
    #     a = self.dist(point_1, point_2)
    #     b = self.dist(point_1, point_3)
    #     c = self.dist(point_3, point_2)
    #
    #     s = (a + b + c) / 2.0
    #     area = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    #     return area
