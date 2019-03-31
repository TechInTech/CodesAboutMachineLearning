#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/14 9:21
# @Author  : Despicable Me
# @Email   : 
# @File    : svm_digits.py
# @Software: PyCharm
# @Explain :

from numpy import *
import matplotlib.pyplot as plt

def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

# 辅助函数2，剪辑后alpha2的解
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif L > aj:
        aj = L
    return aj

def calcB(dataMat, labelMat, alphas):
    m = shape(dataMat)[0]
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).T
    for i in range(m):
        index = 0
        if (alphas[i] > 0) and (labelMatrix[i] > 0):
            index = i
            break
    b1 = zeros((1,1))
    b = labelMatrix[index, :]
    for i in range(m):
        b1 += alphas[i] * labelMatrix[i] * dataMatrix[i,:] * dataMatrix[index,:].T
    b -= b1
    return  b

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n =shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEK(oS, k):
    '''
    计算误差
    :param oS: 数据结构
    :param k:  标号为k的数据
    :return: EK - 标号为k的数据结构
    '''
    # fXK = float(multiply(oS.alphas, oS.labelMat).T *\
    #             (oS.X * oS.X[k, :].T)) + oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)  # 类似第k组数据所得预测标签（未经过激活函数转换）
    EK = fXk - float(oS.labelMat[k])        # 第k组预测结果与实际标签的差值
    return EK     #返回第k组的误差（标量）

def selectJ(i, oS, Ei):
    '''
    内循环启发方式2
    :param i: 标号为i的数据的索引值
    :param oS: 数据结构
    :param Ei: 标号为i的数据误差
    :return: j, maxK - 标号为j或maxK的数据的索引值
             Ej - 标号为j的数据误差
    '''
    maxK = -1
    maxDeltaE = 0
    Ej = 0                                            # 数据初始化
    oS.eCache[i] = [1, Ei]                            # 根据Ei值更新误差缓存
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]   # 返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:                    # 有不为0的误差
        for k in validEcacheList:                     # 遍历,找到最大的Ek
            if k == i:                                # 不计算i,浪费时间
                continue
            Ek = calcEK(oS, k)                        # 计算Ek
            deltaE = abs(Ei - Ek)                     # 计算|Ei-Ek|差的绝对值
            if (deltaE > maxDeltaE):                  # 判断比较差值与预设最大差值的大小
                maxK = k                              # 如果所得差值较大，返回最大值对应的索引值
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej                               # 返回maxK, Ej
    else:                                             # 误差为零
        j = selectJrand(i, oS.m)                      # 随机选择alpha_j的索引值
        Ej = calcEK(oS, j)                            # 计算相应的误差Ej
    return j, Ej                                      # 返回索引j和误差Ej

def updataEK(oS, k):
    '''
    计算误差EK,并更新误差缓存
    :param oS: 数据结构
    :param k:  标号为k的数据
    :return:   无
    '''
    Ek = calcEK(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    '''
    SMO算法的优化部分
    :param i:  标号为i的数据的索引值
    :param oS: 数据结构
    :return: 1 - 有任意一对alpha值发生变化
             0 - 没有任意一对alpha值发生变化或变化太小
    '''
    Ei = calcEK(oS, i)  # 计算误差
    # 优化alpha,设定一定的容错率
    if ((oS.labelMat[i] * Ei < - oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)                            # 使用内循环启发方式2选择alpha_j,并计算Ej
        alphaIold = oS.alphas[i].copy()                       # 保存更新前的alpha值，使用深拷贝(创建新的变量)
        alphaJold = oS.alphas[j].copy()
        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        # eta = 2.0 * oS.X[i,:] * oS.X[j,:].T -oS.X[i,:] * oS.X[i,:].T -oS.X[j,:] * oS.X[j,:].T
        # 步骤3：计算eta
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        # 步骤4：更新alpha_j,所得为未经剪辑的解
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updataEK(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print('j not moving enough')
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] *\
                        (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updataEK(oS, i)
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) *\
        #     oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] *\
        #      (oS.alphas[j,:] - alphaJold) * oS.X[i,:] * oS.X[j,:].T

        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] -\
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]

        # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) *\
        #     oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] *\
        #      (oS.alphas[j] - alphaJold) * oS.X[j,:] * oS.X[j,:].T

        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] -\
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]

        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:  # alpha[i]、alpha[j]的值为0或者C,则它们之间的数都满足KKT条件，此时一般选择它们的中点作为oS.b
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    '''
    :param dataMatIn:  数据集
    :param classLabels: 数据集标签
    :param C:  松弛变量
    :param toler:
    :param maxIter:  最大迭代次数
    :param kTup:    核函数信息元组
    :return:  返回参数b和拉格朗日乘子alpha
    '''
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)  # 定义类
    iter = 0
    entireSet = True
    alphaPairsChanged = 0      # 先定义alphas对未改变
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:         # 全集完整遍历
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 通过innerL选择第二个alpha，并返回alpha改变的次数
            print('fullSet, iter: %d i: %d, pairs changed %d' %\
                      (iter, i, alphaPairsChanged))
            iter += 1
        else:                # 非边界循环
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' %\
                      (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:   # 遍历整个数据集
            entireSet = False
        elif (alphaPairsChanged == 0):  # 未对任意alpha对进行修改
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas

def kernelTrans(X, A, kTup):
    '''
    :param X: 支持向量（数据向量）
    :param A: 数据集中的某一组数据
    :param kTup: 核函数信息的元组
    :return: 计算的核K（列向量），线性核所得结果为<xi,xj>(i=1,2,3,...,n)内积
    '''
    m, n = shape(X)           # 支持向量的维度
    K = mat(zeros((m, 1)))    # 标量K的列向量
    if kTup[0] == 'lin':      # 核函数为线性，只进行内积运算
        K = X * A.T           # 类似于[x1;x2;x3;...;xn] * xi.T = [x1 * xi.T; x2 * xi.T;...; xn * xi.T]
    elif kTup[0] == 'rbf':    # 核函数为径向基
        for j in range(m):
            deltaRow  = X[j,:] - A            # 类似于X(支持向量)中第j行数据元素与数据集中某一行元素做差
            K[j] = deltaRow * deltaRow.T      # 上述元素差的范数的平方
        # 最后得出关于元素差的范数的平方的列向量
        K = exp(K / (-1 * kTup[1]**2))        # 经过高斯径向基核函数运算得到的核K列向量
    else:                                     # 如果核函数不在函数考虑的范围报错
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

def img2vector(filename):
    fr = open(filename)
    returnVect = zeros((1, 1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    '''
    将图片信息转化为矩阵数据形式
    :param dirName: 图片信息所在文件夹
    :return: 图片信息矩阵及其对应的标签矩阵
    '''
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s\%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup = ('rbf', 10)):
    path3 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\kNN\Digits\\trainingDigits'
    path4 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\kNN\Digits\\testDigits'
    dataArr, labelArr = loadImages(path3)
    b, alphas = smoP(dataArr, labelArr, 100, 0.0001, 10000, kTup)
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]           # alpha中大于0的的索引值
    sVs = dataMat[svInd]                       # 支持向量
    labelSV = labelMat[svInd]                  # 支持向量对应的标签
    print('支持向量的个数为：%d' % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('训练误差率为：%f' % (float(errorCount)/m))
    dataArr, labelArr = loadImages(path4)
    errorCount = 0
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('测试误差率为: %f' % (float(errorCount)/m))

def main():
    testDigits(kTup=('rbf', 0.1))

if __name__ == '__main__':
    main()