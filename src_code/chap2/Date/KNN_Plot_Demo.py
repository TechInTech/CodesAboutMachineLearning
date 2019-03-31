#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 20:34
# @Author  : Despicable Me
# @Site    : 
# @File    : KNN_Plot_Demo.py
# @Software: PyCharm
# @Explain :

# 绘制原始数据2-D图

from KNN import *

import matplotlib.pyplot as plt
plt.style.use('ggplot')

dating_data, dating_label = file2matrix('datingTestSet.txt')
# 将数据按类别划分
type1_x = []                                        # 类别1
type1_y = []

type2_x = []                                        # 类别2
type2_y = []

type3_x = []                                        # 类别3
type3_y = []

for i in range(len(dating_label)):
    if dating_label[i] == 1:                        # 如果第i行标签为类别1
        type1_x.append(dating_data[i][0])
        type1_y.append(dating_data[i][1])
    elif dating_label[i] == 2:                      # 如果第i行标签为类别2
        type2_x.append(dating_data[i][0])
        type2_y.append(dating_data[i][1])
    elif dating_label[i] == 3:                      # 如果第i行标签为类别3
        type3_x.append(dating_data[i][0])
        type3_y.append(dating_data[i][1])

plt.figure(figsize=(10, 5))

type1 = plt.scatter(type1_x, type1_y, c = 'red', label = 'DidntLike')
type2 = plt.scatter(type2_x, type2_y, c = 'brown', label = 'SmallDoses')
type3 = plt.scatter(type3_x, type3_y, c = 'lime', label = 'LargeDoses')

# plt.xlabel('Fling In Air')
# plt.ylabel('Playing Game')

plt.grid(True)

plt.legend(loc = (1,0))


plt.show()