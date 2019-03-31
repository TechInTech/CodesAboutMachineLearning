#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019/1/3 10:44
# @Author  : Despicable Me
# @Email   : 
# @File    : pca.py
# @Software: PyCharm
# @Explain :
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import math


def loadDataSet(fileName, delim= '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat= 9999999):
    # cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
    '''
    方差：（一维）度量两个随机变量关系的统计量
    协方差： （二维）度量各个维度偏离其均值的程度
    协方差矩阵：（多维）度量各个维度偏离其均值的程度

    当 cov(X, Y)>0时，表明X与Y正相关；(X越大，Y也越大；X越小Y，也越小。这种情况，我们称为“正相关”。)
    当 cov(X, Y)<0时，表明X与Y负相关；
    当 cov(X, Y)=0时，表明X与Y不相关。
    '''
    """
    PCA的理念就是将高维空间的数据映射到低维空间，降低维度之间的相关性，并且使自身维度的方差尽可能的大;
    协方差矩阵对角线上是维度的方差，其他元素是两两维度之间的协方差(即相关性);
    PCA的目的之一：降低维度之间的相关性，也就说减小协方差矩阵非对角线上的值。如何减小呢？可以使协方差矩阵变成对角矩阵。
    对角化后的矩阵，其对角线上是协方差矩阵的特征值
    """

    meanVals = mean(dataMat, axis= 0)                 # 各个特征的均值
    meanRemoved = dataMat - meanVals                  # 特征值减去均值
    convMat = cov(meanRemoved, rowvar= 0)             # 协方差矩阵
    eigVals, eigVects = linalg.eig(mat(convMat))      # 协方差矩阵的特征值与特征向量
    eigValInd = argsort(eigVals)                      # 对特征值进行升序排列，返回索引值
    eigValInd = eigValInd[:-(topNfeat + 1):-1]        # 由排序后的逆序得出topNfeat个最大的特征向量的索引
    redEigVects = eigVects[:, eigValInd]              # 得出topNfeat个最大的特征向量
    lowDDataMat = meanRemoved * redEigVects           #
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    path1 = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\\13.PCA\secom.data'
    datMat = loadDataSet(path1, ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])    # 计算所有非NaN数据的平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal           # 将所有NaN项替换为平均值
    return datMat


def main():
    # path = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\\13.PCA\\testSet.txt'
    # dataMat = loadDataSet(path)
    # print(shape(dataMat), type(dataMat))
    # lowDMat, reconMat = pca(dataMat, 1)
    # print(shape(lowDMat))
    #
    # # # **************************** 绘制分布图 ***********************

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker= '^', s= 90)
    # ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker= 'o', s= 5, c= 'red')
    # ax.axis([5, 14, 3, 15])
    # plt.show()

    # # *************************** 通过PCA对半导体制造数据降维 *********
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis= 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar= 0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    print('协方差矩阵的特征值：','\n', eigVals, '\n', '协方差矩阵的特征向量：', '\n', eigVects)
    eigvalInd = argsort(eigVals)
    topNfeat = 20
    eigvalInd = eigvalInd[:-(topNfeat + 1):-1]
    cov_var_total = float(sum(eigVals))        # 方差的总值

    # # ******* 绘制前20个主成分占总方差的百分比 **************
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(arange(1, topNfeat + 1), eigVals[eigvalInd]/cov_var_total*100, marker='o')
    y_max = math.ceil(max(eigVals[eigvalInd]/cov_var_total*100))
    ax.tick_params(direction= 'in')
    ax.axis([0, topNfeat, 0, y_max])
    plt.xlabel('主成分数目', fontproperties= 'Kaiti', fontsize= 10)
    plt.ylabel('方差的百分比', fontproperties= 'SimHei', fontsize= 10)
    plt.show()
    # # **********************

    # # ***************** 前20个主成分所占百分比及累计百分比 ******************
    Accu_cov_var = 0
    for i in range(topNfeat):
        single_cov_var = float(eigVals[eigvalInd[i]])       # 第i个主成分的方差
        Accu_cov_var += single_cov_var                      # 前i个主成分的方差累计和
        print('主成分:{:2}'.format(i+1),'方差占比:{:4.2%}'.format(single_cov_var/cov_var_total),\
              '累计方差占比:{:4.2%}'.format(Accu_cov_var/cov_var_total))

if __name__ == '__main__':
    main()