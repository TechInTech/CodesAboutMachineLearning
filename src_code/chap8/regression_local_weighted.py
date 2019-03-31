#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/21 11:35
# @Author  : Despicable Me
# @Email   : 
# @File    : regression_local_weighted.py
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

def lwlr(testPoint, xArr, yArr, k=1.0):
    '''
    计算局部加权线性回归函数: 给定x空间中任意一点，计算对应的预测值yHat
    :param testPoint:   测试数据
    :param xArr:        选定的输入数据集
    :param yArr:        选定的输出数据集
    :param k:           核函数中的参数
    :return:            测试数据的预测输出
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))                                      # 局部加权矩阵初始化
    for j in range(m):
        diffMat = testPoint - xMat[j,:]                          # 测试数据与选定数据的差值
        weights[j,j] = exp(diffMat * diffMat.T/(-2.0 * k**2))    # 局部加权矩阵
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))                     # 回归系数
    return testPoint * ws                                        # 返回预测值(列表与矩阵相乘)

def lwlrTest(testArr, xArr, yArr, k=1.0):
    '''
    对给定的数据点求出对应的预测值
    :param testArr:     待进行预测的测试集输入矩阵
    :param xArr:        选定的(训练)数据集输入
    :param yArr:        选定的(训练)数据集输出
    :param k:           核函数中的参数
    :return: yHat    -  测试数据集输入的预测输出
    '''
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    '''
    计算预测误差的大小
    :param yArr:     实际值
    :param yHatArr:  预测值
    :return:         预测误差
    '''
    return ((yArr - yHatArr)**2).sum()

def LocalDatashow(xArr, yArr, K):
    '''
    绘制由不同的核函数参数k所得到的不同回归系数ws下的回归曲线图
    :param xArr:    输入数据集
    :param yArr:    输出数据集
    :param ws:      不同回归系数的列表
    :return:        无
    '''
    m = len(K)
    yHat = []
    for index in K:
        y_Pred = lwlrTest(xArr, xArr, yArr, index)
        yHat.append(y_Pred)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    plt.title('LWLR of different K')
    plt.axis('off')      # 使外部大图像的坐标轴不显示，防止与小图像的坐标轴重叠

    for i in range(1, m + 1):
        ax = fig.add_subplot(m * 100 + 10 + i)
        ax.plot(xSort[:,1], yHat[i -1][srtInd], c= 'green', linewidth= 2, label= 'k='+str(K[i-1]))
        ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s= 2, c= 'red')
        plt.axis([0.0, 1.0, 3.0, 5.0])
        plt.legend(loc= 'upper left')

    plt.show()

def main():
    path1 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression\ex0.txt'
    dataMat, labelMat = loadDataSet(path1)  # 加载数据

    # ********************** 局部加权线性回归 ******************************************
    dataMat_Pred = lwlr(dataMat[0], dataMat, labelMat, 1.0)
    print('输入', labelMat[0], '的预测值为', dataMat_Pred.flatten().A[0][0])
    # yHat = lwlrTest(dataMat, dataMat, labelMat, 0.01)
    # print(yHat)
    k = [1.0, 0.01, 0.003]
    LocalDatashow(dataMat, labelMat, k)   # 展示不同参数k下的回归曲线
    # ********************************************************************************

    # # ************************ 分析预测误差的大小(预测鲍鱼的年龄) ****************************************
    # path2 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression\\abalone.txt'
    # abX, abY = loadDataSet(path2)  # 导入鲍鱼数据
    # # 预测输出
    # yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)  # 参数k为0.1时的预测输出
    # yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)  # 参数k为1时的预测输出
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)  # 参数k为10时的预测输出
    # # 计算(训练集的)预测误差
    # rss01 = rssError(abY[0:99], yHat01.T)
    # rss1 = rssError(abY[0:99], yHat1.T)
    # rss10 = rssError(abY[0:99], yHat10.T)
    # print('参数K在三种情况下(训练集)的预测误差：')
    # print('k = 0.1: %.4f' % rss01)
    # print('k = 1: %.4f' % rss1)
    # print('k = 10: %.4f' % rss10)
    # # *********
    # print('********************\n********************')
    # # *********
    # # 预测输出
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)  # 参数k为0.1时的预测输出
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)  # 参数k为1时的预测输出
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)  # 参数k为10时的预测输出
    # # 计算(测试集的)预测误差
    # rss01 = rssError(abY[100:199], yHat01.T)
    # rss1 = rssError(abY[100:199], yHat1.T)
    # rss10 = rssError(abY[100:199], yHat10.T)
    # print('参数K在三种情况下(测试集)的预测误差：')
    # print('k = 0.1: %.4f' % rss01)
    # print('k = 1: %.4f' % rss1)
    # print('k = 10: %.4f' % rss10)

if __name__ == '__main__':
    main()