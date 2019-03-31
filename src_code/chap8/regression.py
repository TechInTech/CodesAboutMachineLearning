#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/16 15:55
# @Author  : Despicable Me
# @Email   : 
# @File    : regression.py
# @Software: PyCharm
# @Explain :

from numpy import *
import matplotlib.pyplot as plt
import matplotlib

def loadDataSet(fileName):
    '''
    将文本文件信息转化为输入数据和输出数据
    :param fileName:     文本文件名
    :return:  dataMat  - 输入数据(列表)
              labelMat - 输出数据(列表）
    '''
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    '''
    标准线性回归函数，计算回归系数
    :param xArr:  输入数据
    :param yArr:  输出数据
    :return: ws - 回归系数
    '''
    xMat = mat(xArr)                # 将数据转化为矩阵形式
    yMat = mat(yArr).T              #
    xTx = xMat.T * xMat             # 求X^(T)*X
    if linalg.det(xTx) == 0.0:      # 如果xTx的行列式为零，输出警告，矩阵不可逆
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)    # 计算回归系数
    return ws

def rssError(yArr, yHatArr):
    '''
    计算预测误差的大小
    :param yArr:     实际值
    :param yHatArr:  预测值
    :return:         预测误差
    '''
    return ((yArr - yHatArr)**2).sum()

def Datashow(xArr, yArr, ws):
    '''
    绘制原始数据分布图和回归直线
    :param xArr:     输入数据
    :param yArr:     实际输出数据
    :param ws:       回归系数
    :return:         无
    '''
    xMat = mat(xArr)          # 数据转化为矩阵形式
    yMat = mat(yArr)          #

    xCopy = xMat.copy()       # 在创建新变量，存储输入数据
    xCopy.sort(0)             # 将输入数据按列升序排列
    yHat = xCopy * ws         # 计算预测输出

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # SValue = 5 * ones((shape(xMat)[0], 1))               # 散点图中数据显示的大小，数量等于散点图中输入的个数
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s= 5, c= 'green')   # 原始数据散点图
                     # xMat[:, 1].flatten() 返回将xMat的第1列元素展开为1维数组，原数矩阵不受影响;.A为将矩阵转化为数组
                     # 参数s指定数据显示的大小

    ax.plot(xCopy[:, 1], yHat, c= 'red', linewidth= 2)                               # 回归直线
    plt.axis([-0.2, 1.2, 2.5, 5.0])                                                  # 坐标轴的约束
    plt.xlabel('输入', fontproperties= 'STSong', fontsize= 12)
    plt.ylabel('输出', fontproperties= 'STSong', fontsize= 12)
    plt.show()

def main():
    path1 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression\ex0.txt'
    dataMat, labelMat = loadDataSet(path1)    # 加载数据

    # *********************** 线性回归函数 ********************************************
    ws = standRegres(dataMat, labelMat)       # 计算回归系数
    Datashow(dataMat, labelMat, ws)           # 绘制原始数据的散点图和回归直线图
    xMat = mat(dataMat)
    yMat = mat(labelMat)
    yHat = xMat * ws
    corrmatrix = corrcoef(yHat.T, yMat)       # 计算回归系数矩阵
    print('回归预测值与实际值之间的协方差矩阵:', '\n', corrmatrix)
    # ********************************************************************************

if __name__ == '__main__':
    main()