# !/usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/6 14:25
# @Author  : Despicable Me
# @Email   :
# @File    : Pegasos.py
# @Software: PyCharm
# @Explain : SVM的Pegasos(原始估计梯度求解器)算法

from numpy import *
import matplotlib.pyplot as plt
import math


def predict(w, x):
    return w * x.T


def batchPegasos(dataSet, labels, lam, T, k):
    """
    Pegasos算法的串行版本
    :param dataSet:  数据集
    :param labels:   类别标签
    :param lam:      固定值
    :param T:        迭代次数
    :param k:        待处理列表的大小
    :return: w       回归系数
    """
    m, n = shape(dataSet)
    w = zeros(n)
    dataindex = list(range(m))        # 建立数据集大索引列表
    for t in range(1, T + 1):
        wdelta = mat(zeros(n))        # T次迭代中，每次需要重新计算eta
        eta = 1.0/(lam * t)           # eta为学习率，代表了权重调整幅度的大小(也可以理解为随机梯度的步长，使它不断减小，便于拟合)
        random.shuffle(dataindex)     # 将序列的所有元素随机排序
        for j in range(k):
            i = dataindex[j]          # 随机获得数据行索引
            p = predict(w, dataSet[i, :])                # 得到第i行的预测结果
            if labels[i] * p < 1:
                wdelta += labels[i] * dataSet[i, :].A    # 将分类错误的值全部累加之后更新权重向量
        # w通过不断的随机梯度的方式来优化
        w = (1.0 - 1/t) * w + (eta/k) * wdelta
    return w


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def DataShow(datamat, labellist, w):
    x1 = []; y1 = []             # 类别-1
    x2 = []; y2 = []             # 类别1
    for i in range(len(labellist)):
        if labellist[i] == -1.0:
            x1.append(datamat[i, 0])
            y1.append(datamat[i, 1])
        else:
            x2.append(datamat[i, 0])
            y2.append(datamat[i, 1])
    xmin = min(min(x1), min(x2))
    xmax = max(max(x1), max(x2))
    ymin = min(min(y1), min(y2))
    ymax = max(max(y1), max(y2))
    m = shape(datamat)[0]
    x = arange(xmin, xmax, (xmax - xmin)/m)
    y = -(w[0,0] * mat(x).T + 0.0)/w[0,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, c= 'blue', marker= 'o', label= 'class of -1')
    ax.scatter(x2, y2, c= 'red', marker= '*', label= 'class of 1')
    ax.plot(x, y.flatten().A[0], 'g--', linewidth= 2)
    ax.axis([math.ceil(xmin - 1), math.ceil(xmax), math.ceil(ymin - 1), math.ceil(ymax)])
    plt.legend(loc= 'upper left')
    plt.show()



def main():
    path = 'D:\Projects_Python\Dong\MachineLearningInAction\dataSet\\15.BigData_MapReduce\\testSet.txt'
    dataArr, labelArr = loadDataSet(path)
    dataMat = mat(dataArr)
    print(shape(dataMat))
    para_w = batchPegasos(dataMat, labelArr, 1, 50, 100)
    # print(para_w)
    DataShow(dataMat, labelArr, para_w)


if __name__ == '__main__':
    main()