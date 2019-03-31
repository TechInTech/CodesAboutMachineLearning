#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/24 21:20
# @Author  : Despicable Me
# @Email   : 
# @File    : try.py
# @Software: PyCharm
# @Explain :

# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:19:51 2018
@author: lizihua
"""
from numpy import *
from tkinter import *

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# 加载数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 在给定的特征和特征值的情况下，通过数组过滤的方式将上述数据分成二个子集返回
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 创建叶结点，此时数据不能继续切分
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

def isTree(obj):                           # 判断对象是否为叶子节点
    return (type(obj).__name__ == 'dict')  # 不为叶子节点返回1

# 创建
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# errType：计算总方差（平方误差和）函数 = regErr
# ops：用户定义的参数构成的元组，用来完成树的构建，
# ops=(tolS,tolN),tolS:容许的误差下降值；tolN：切分的最小样本
# chooseBestSplit的目的是找到数据的最佳二元切分方式，若无，则返回None,并同时调用createTree产生叶结点
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0];
    tolN = ops[1]
    # 停止切分的条件1：若剩余的不同特征数目=1时，则退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf;
    bestIndex = 0;
    bestValue = 0
    for featIndex in range(n - 1):
        # for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
        for splitVal in dataSet[:, featIndex]:
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 当切分的数据集小于切分的最小样本tolN时,则退出循环
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 停止切分的条件2：若误差减小在容许下降误差值tolS内，则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 停止切分的条件3：当切分的数据集小于切分的最小样本tolN时，则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 创建树：递归函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 树回归
# 线性模型函数，将被以下两个函数调用，其余过程与简单的线性回归函数过程一般
def linearSolve(dataSet):
    m, n = shape(dataSet)
    # 初始化X,Y
    X = mat(ones((m, n)));
    Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1];
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 生成叶结点模型
def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


# 线性模型误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


def reDraw(tolS, tolN):
    reDraw.f.clf()  # clear the figure
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2: tolN = 2
        myTree = createTree(reDraw.rawDat, modelLeaf, modelErr, (tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat, modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:, 0].A, reDraw.rawDat[:, 1].A, s=5)  # use scatter for data set
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)  # use plot for yHat
    reDraw.canvas.show()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()  # get values from Entry boxes
    reDraw(tolS, tolN)


root = Tk()
# 用画布来替换绘制占位符，并删掉对应标签，即下面的Plot Place Holder那句
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
# 待删除，只是代替图片占一个位置
# Label(root,text='Plot Place Holder').grid(row = 0, columnspan = 3)
# tolN
Label(root, text='tolN').grid(row=1, column=0)
tolNentry = Entry(root)  # Entry：单行文本输入框
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')  # 默认值为10
# tolS
Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')  # 默认值为1.0
# 按钮
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)
# 按钮整数值
chkBtnVar = IntVar()
# 复选按钮
chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)
# 初始化数据
reDraw.rawDat = mat(loadDataSet('D:\Projects_Python\Dong\GitHub Files\AiLearning\db\9.RegTrees\sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)
root.mainloop()
