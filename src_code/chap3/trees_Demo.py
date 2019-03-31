#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/25 21:00
# @Author  : Despicable Me
# @Site    : 
# @File    : trees_Demo.py
# @Software: PyCharm
# @Explain :

from trees import *
from treePlotter import *

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

myData, labels = createDataSet()

shannon = calcShannonEnt(myData)

print('the shannonEnt is:', shannon)

list1 = splitDataSet(myData, 0, 1)
list2 = splitDataSet(myData, 0, 0)

print(list1, '\n', list2)

Bestfeature = chooseBestFeatureToSplit(myData)

print('The best choose is the', Bestfeature, 'feature!')

# myTree = createTree(myData, labels)
#
# print(myTree)

mytree = retrieveTree(0)

re1 = classify(mytree, labels, [1, 1])
print(re1)

createPlot(mytree)

storeTree(mytree, 'D:\Projects_Python\DataSets\machinelearninginaction\Ch03\classifierStorage.txt')
re3 = grabTree('D:\Projects_Python\DataSets\machinelearninginaction\Ch03\classifierStorage.txt')
print(re3)

fr = open('D:\Projects_Python\DataSets\machinelearninginaction\Ch03\lenses.txt', 'r')
print(fr)

lenses = [inst.strip().split('\t') for inst in fr.readlines()]

lensesLabels = ['age', 'prescript', 'astimatic', 'tearRate']

lensesTree = createTree(lenses, lensesLabels)

print(lensesTree)

createPlot(lensesTree)