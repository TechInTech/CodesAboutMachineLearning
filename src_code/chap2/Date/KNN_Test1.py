#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 16:27
# @Author  : Despicable Me
# @Site    : 
# @File    : KNN.py
# @Software: PyCharm
# @Explain :

#  K-近邻算法

from numpy import *
from KNN import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# def createDataSet():
#     group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels
#
# group, labels = createDataSet()
# numA = 0
# numB = 0
# for i in labels:
#     if i == 'A':
#         numA += 1
#     elif i == 'B':
#         numB += 1
# print('The number of class A:', numA)
# print('The number of class B:', numB)

datingDataMat, dating_label = file2matrix('datingTestSet.txt')

print(datingDataMat)

dating_data, ranges, minvals = autoNorm(datingDataMat)

print(dating_data)

type1_x = []                                        # 类别1
type1_y = []
type1_z = []

type2_x = []                                        # 类别2
type2_y = []
type2_z = []

type3_x = []                                        # 类别3
type3_y = []
type3_z = []

for i in range(len(dating_label)):
    if dating_label[i] == 1:                        # 如果第i行标签为类别1
        type1_x.append(dating_data[i][0])
        type1_y.append(dating_data[i][1])
        type1_z.append(dating_data[i][2])
    elif dating_label[i] == 2:                      # 如果第i行标签为类别2
        type2_x.append(dating_data[i][0])
        type2_y.append(dating_data[i][1])
        type2_z.append(dating_data[i][2])
    elif dating_label[i] == 3:                      # 如果第i行标签为类别3
        type3_x.append(dating_data[i][0])
        type3_y.append(dating_data[i][1])
        type3_z.append(dating_data[i][2])

# datingClassTest()
# classifyPerson()
handwritingClassTest()

plt.figure()
plt.title('Comparison Of Fling and Playing')
ax1 = plt.scatter(type1_x, type1_y, c = 'lime')
ax2 = plt.scatter(type2_x, type2_y, c = 'green')
ax3 = plt.scatter(type3_x, type3_y, c = 'yellow')

plt.legend((ax1, ax2, ax3), ('DidntLike', 'SmallDoses', 'largeDoses'), loc = (1, 0))

plt.figure()
plt.title('Comparison Of Playing and Eating Ice Cream')

plt.scatter(type1_y, type1_z, c = 'red', label = 'DidntLike')
plt.scatter(type2_y, type2_z, c = 'green', label = 'SmallDoses')
plt.scatter(type3_y, type3_z, c = 'blue', label = 'largeDoses')

plt.legend(loc = (1, 0))

plt.show()