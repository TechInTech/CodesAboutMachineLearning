#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 16:54
# @Author  : Despicable Me
# @Site    :
# @File    : KNN.py
# @Software: PyCharm
# @Explain :

from numpy import *
import os, sys

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 将文件转化为numpy矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()

    numberOfLines = len(arrayLines)
    returnMat = zeros((numberOfLines, 3))
    classlabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listfromline = line.split('\t')                      # 采用Tab将line分为列表
        returnMat[index, :] = listfromline[0:3]
        labels = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
                                                             # 将对应的标称标签转换为整数表示，常采用字典
        classlabelVector.append(labels[listfromline[-1]])
        index += 1
    return returnMat, classlabelVector

# KNN分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                           # 数据集样本大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet          # 将待分类样本与已给定样本进行差值运算
    sqDiffMat = diffMat**2                                   # 差的平方
    sqDistances = sqDiffMat.sum(axis=1)                      # 距离的平方
    distances = sqDistances**0.5                             # 所求距离
    sortedDistIndicies = distances.argsort()                 # 返回数组值从大到小的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), \
          key = lambda item:item[1], reverse= True)
    return sortedClassCount[0][0]

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], \
                                     datingLabels[numTestVecs:m], 4)
        print('the classifier came back with: %d, the right result is: %d' \
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print('The total error rate is:', (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArray = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArray - minVals)/ranges, normMat,datingLabels, 4)
    print('You will probably like this person:', resultList[classifierResult - 1])

def img2vector(filename):
    fr = open(filename)
    returnVect = zeros((1, 1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    relative_path1 = r'D:\Projects_Python\DataSets\machinelearninginaction\Ch02\digits\trainingDigits'
    relative_path2 = r'D:\Projects_Python\DataSets\machinelearninginaction\Ch02\digits\testDigits'
    hwlabels = []
    trainingDatalist = os.listdir(relative_path1)
    numlist = len(trainingDatalist)
    trainingMat = zeros((numlist, 1024))
    for i in range(numlist):
        fileNameStr = trainingDatalist[i]
        fileName = fileNameStr.split('.')[0]
        classNumStr = int(fileName.split('_')[0])
        hwlabels.append(classNumStr)
        trainingMat[i,:] = img2vector(relative_path1 + '\%s' % fileNameStr)
    TestDatalist = os.listdir(relative_path2)
    errorCount = 0.0
    numtest = len(TestDatalist)
    for i in range(numtest):
        fileNameStr = TestDatalist[i]
        fileName = fileNameStr.split('.')[0]
        classNumStr = int(fileName.split('_')[0])
        testMat = img2vector(relative_path2 + '\%s' % fileNameStr)
        classifierresult = classify0(testMat, trainingMat, hwlabels, 4)

        print('the classifier came back with: %d, \
                the real answer is: %d' % (classifierresult, classNumStr))
        if (classifierresult != classNumStr):
            errorCount += 1.0
    print('the total number of error is:', errorCount)
    print('the total rate is: %f', float(errorCount/numtest))
