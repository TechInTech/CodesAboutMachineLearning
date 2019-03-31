#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019/1/3 22:00
# @Author  : Despicable Me
# @Email   : 
# @File    : svdRec.py
# @Software: PyCharm
# @Explain :
from numpy import *
from numpy import linalg as la


def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


def loadExData2():
    return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
            [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
            [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


def ecludSim(inA, inB):
    """
    求向量inA、inB的二范数(欧氏距离计算相似度)
    """
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    """
    皮尔逊相关系数,计算相似度
    """
    if len(inA) < 3:
        """
        皮尔逊相关系数法，会检查向量中是否存在3个或更多的点，如果不存在，
        返回相关系数为1。因为两个向量完全相关
        """
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar= 0)[0][1]


def cosSim(inA, inB):
    """
    余弦相似度
    """
    num = float(inA.T * inB)    # 分子
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num/denom)


def standEst(dataMat, user, simMeas, item):
    """
    计算在给定相似度计算方法的条件下，用户对物品的估计评分值
    :param dataMat:   数据集
    :param user:      用户编号
    :param simMeas:   相似度计算方法
    :param item:      未评分物品编号
    :return:          对未评分物品的预测得分
    """
    n = shape(dataMat)[1]          # 物品种类
    simTotal = 0.0                 # 总的相似度
    ratSimTotal = 0.0              #
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]  # 给出两个物品当中已经被评分的哪些元素
        if len(overLap) == 0:       # 如果两个物品没有任何重合元素，相似度为0
            similarity = 0
        else:                       # 如果两个物品存在重合元素，则基于这些重合元素计算相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity                    # 相似度不断累加
        ratSimTotal += similarity * userRating    # 相似度和当前用户评分的乘积的累加
    if simTotal == 0:
        return 0
    else:                                         # 通过除以所有的评分总和，对相似度评分的乘积进行归一化，
        return ratSimTotal/simTotal               # 这样就保证了最后的评分值在0到5之间


def recommend(dataMat, user, N= 3, simMeas= cosSim, estMethod= standEst):
    """
    产生最高的N个推荐结果
    :param dataMat:    数据集
    :param user:       给定的用户
    :param N:          默认值为3
    :param simMeas:    相似度计算方法
    :param estMethod:  估计方法
    :return:
    """
    unratedItems = nonzero(dataMat[user,:].A==0)[1]      # 对给定的用户建立一个为评分的物品列表
    if len(unratedItems) == 0:                           # 如果不存在为评分物品，返回
        return 'you rated everything'
    itemScores = []                                      #
    for item in unratedItems:                            # 在所有的未评分物品上进行循环
        estimatedScore = estMethod(dataMat, user, simMeas, item)      # 得到该物品的预测得分
        itemScores.append((item, estimatedScore))                     # 该物品的编号和估计得分值放在一个元素列表
    return sorted(itemScores, key= lambda jj:jj[1],reverse=True)[:N]  # 按得分对该列表进行从大到小的排序


def svdEst(dataMat, user, simMeas, item):
    """
    基于SVD的评分估计：对给定用户给定物品构建了一个评分估计值
    """
    n = shape(dataMat)[1]
    simTotal = 0.0                  # 相似度总和初始化
    ratSimToal = 0.0                # 相似度的乘积的初始化
    U, Sigma, VT = la.svd(dataMat)  # 将数据进行svd(奇异值)分解
    Sig4 = mat(eye(6) * Sigma[:6])  # 构建一个对角矩阵
    xformedItems = dataMat.T * U[:, :6] * Sig4.I   # 利用U矩阵将物品装换到低维空间
    for j in range(n):                             #
        userRate = dataMat[user, j]                # 第user个用户对j物品的评分
        if userRate == 0 or j == item:             # 如果
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)  # 在用户对应行的所有元素上进行遍历
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimToal += similarity * userRate
    if simTotal == 0:
        return 0
    else:
        return ratSimToal/simTotal


def printMat(inMat, thresh= 0.8):
    """打印矩阵，通过阈值来界定"""
    mat_outer = []
    for i in range(32):
        mat_inner = []
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                # print (1,)
                mat_inner.append(1)
            else:
                # print (0,)
                mat_inner.append(0)
        # print('')
        mat_outer.append(mat_inner)
    print(array(mat_outer))


def imgCompress(numSV= 3, thresh= 0.8):
    """实现图像的压缩，基于任意给定的奇异值数目来重构图像"""
    """
    numSV   任意给定的奇异值数目
    thresh  给定的阈值
    """
    path = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\\14.SVD\\0_5.txt'
    myl = []
    for line in open(path).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print('**** original matrix ****')
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:,:numSV] * SigRecon * VT[:numSV,:]
    print('**** reconstructed matrix using %d singular values ****' % numSV)
    printMat(reconMat, thresh)


def Test1():
    """
    测试奇异值分解函数svd
    """
    data = loadExData()                                                    # 原始矩阵
    U, Sigma, VT = la.svd(data)                                        # 奇异值分解， sigma为奇异值对角矩阵
    print(Sigma)
    Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    recon_ori_matrix = U[:,:3] * Sig3 * VT[:3,:]                           # 重构原始矩阵
    print(recon_ori_matrix.A)


def Test2():
    """
    几种相似度方法的比较
    """
    myMat = mat(loadExData())
    eclu_Sim = ecludSim(myMat[:,0], myMat[:,4])      # 欧氏距离法计算相似度
    pears_Sim1 = pearsSim(myMat[:,0], myMat[:,0])     # 皮尔逊相关系数(相似度)
    pears_Sim2 = pearsSim(myMat[:,0], myMat[:,4])
    cos_Sim = cosSim(myMat[:,0], myMat[:,0])         # 余弦相似度
    print('欧几里德相似度：%f' % eclu_Sim, '\n', '皮尔逊相关系数1：%f' % pears_Sim1, '\n',\
          '皮尔逊相关系数2：%f' % pears_Sim2, '\n', '余弦相似度：%f' % cos_Sim)


def Test3():
    """
    基于物品相似度的推荐引擎
    """
    myMat = mat(loadExData())
    myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
    myMat[3,3] = 2
    print(myMat)
    recom1 = recommend(myMat, 2)
    print(recom1)
    recom2 = recommend(myMat, 2, simMeas= ecludSim)
    print(recom2)
    recom3 = recommend(myMat, 2, simMeas= pearsSim)
    print(recom3)


def Test4():
    """
    利用SVD提高推荐的效果
    """
    myMat = mat(loadExData2())
    U, Sigma, VT = la.svd(myMat)
    print('特征值：', '\n', Sigma)
    print('奇异值的总能量：%f' % sum(Sigma**2))       # 奇异值的总能量为特征值的平方和
    print('奇异值90%的总能量:{:f}'.format(sum(Sigma**2) * 0.9))
    Acc_Chara = 0
    for i in range(len(Sigma)):
        Acc_Chara += Sigma[i]**2
        print('前%d个奇异值的总能量:%f' % (i+1, Acc_Chara))


def Test5():
    myMat =  mat(loadExData2())
    recom = recommend(myMat, 1, estMethod= svdEst)
    print('评分结果：', '\n', recom)


def Test6():
    imgCompress(2)


if __name__ == '__main__':
    # Test1()
    # Test2()
    # Test3()
    # Test4()
    # Test5()
    Test6()