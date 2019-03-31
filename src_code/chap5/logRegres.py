#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 22:27
# @Author  : Despicable Me
# @Site    : 
# @File    : logRegres.py
# @Software: PyCharm
# @Explain :

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    path1 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Logistic\\testSet.txt'
    dataMat = []
    labelMat = []
    fr = open(path1)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return longfloat(1.0 / (1 + exp( -inX)))

# 原始的梯度上升法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.T * error
    return weights

# 绘制分布图以及回归曲线
def plotBestFit(wei):
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c= 'green', marker = 'o')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y.T)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    Coeffedata = []
    # print(weights)
    # for j in range(200):
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error *dataMatrix[i]
        Coeffedata.append(weights)
            # print(h, weights, error)
    return weights, Coeffedata

# 绘制回归系数的收敛曲线
def plotTheCoeff(dataOfCoe):
    m, n = shape(dataOfCoe)
    y0 = []
    y1 = []
    y2 = []
    for i in range(m):
        y0.append(dataOfCoe[i][0])
        y1.append(dataOfCoe[i][1])
        y2.append(dataOfCoe[i][2])
    x = range(m)
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(x, y0)
    plt.ylabel('x0')
    # plt.axis([0, m, -8, 1])
    ax2 = fig.add_subplot(312)
    ax2.plot(x, y1)
    plt.ylabel('x1')
    # plt.axis([0, m, -1 * 1.2, 1])
    ax3 = fig.add_subplot(313)
    ax3.plot(x, y2)
    plt.ylabel('x2')
    # plt.axis([0, m, -1.5, 1.4])
    plt.xlabel('Time of Iter')

    plt.show()

# 改进的随机梯度上升法
def stocGradAscent1(dataMatrix, classLabels, numIter = 40):
    m, n = shape(dataMatrix)
    weights = ones(n)      # weights为(1,n)数组
    Coeffedata = []
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (i + j + 1) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))    # 对应元素相乘之后再相加，即点乘
            error = array(classLabels[randIndex], dtype= 'float64') - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
            Coeffedata.append(weights)
    # return weights, Coeffedata
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0



def colicTest():
    path2 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Logistic\horseColicTraining.txt'
    path3 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Logistic\horseColicTest.txt'
    frTrain = open(path2)
    frTest = open(path3)
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(currLine[21])
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, numIter= 600)
    errorCount = 0.0
    numTestCount = 0.0
    for line in frTest.readlines():
        numTestCount += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestCount
    print('the error rate is: %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d times iterations the average error \
    rate is: %f' % (numTests, errorSum / float(numTests)))