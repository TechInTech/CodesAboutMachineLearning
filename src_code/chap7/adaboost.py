#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/14 10:41
# @Author  : Despicable Me
# @Email   :
# @File    : adaboost.py
# @Software: PyCharm
# @Explain :

from numpy import *
import matplotlib.pyplot as plt
import matplotlib
def loadSimpData():
    dataMat = matrix([[1., 2.1],
                      [1.5, 1.6],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels
def showData(dataMat, classLabels):
    '''
    展示数据集的分布
    :param dataMat:        数据集矩阵
    :param classLabels:    数据集对应的标签列表
    :return:               数据集在二维平面的分布图
    '''
    m, n = shape(dataMat)
    data_plus = []
    data_minus = []
    for i in range(m):
        if classLabels[i] == 1.0:
            data_plus.append(dataMat[i])
        elif classLabels[i] == -1.0:
            data_minus.append(dataMat[i])
    data_plus_Arr = array(data_plus).T
    data_minus_Arr = array(data_minus).T
    plt.figure()
    plt.scatter(data_plus_Arr[0], data_plus_Arr[1], color = 'red', marker= '*', label = 'class of 1')
    plt.scatter(data_minus_Arr[0], data_minus_Arr[1], color = 'green', marker = 'o', label = 'class of -1')
    plt.legend(loc = (0,1))
    plt.show()

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类
    :param dataMatrix:  输入数据矩阵
    :param dimen:       第dimen列特征
    :param threshVal:   阈值
    :param threshIneq:
    :return:            分类后的列向量
    '''
    retArrays = ones((shape(dataMatrix)[0], 1))     # 初始化为全1
    if threshIneq == 'lt':
        retArrays[dataMatrix[:, dimen] <= threshVal] = -1.0    # 将第dimen个特征中小于阈值的行对应的值置为-1
    else:
        retArrays[dataMatrix[:, dimen] > threshVal] = -1.0     # 将第dimen个特征中大于阈值的行对应的值置为-1
    return retArrays

def buildStump(dataArr, classLabels, D):
    '''
    找到数据集上最佳的单层决策树
    :param dataArr:          数据矩阵
    :param classLabels:      数据对应的类别标签
    :param D:                样本权重向量
    :return:  bsetStump   -  最佳决策树信息字典
              minError    -  最小误差
              bestClasEst -  最佳类估计
    '''
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n =shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = float('inf')                                     # 最小误差初始化为正无穷大
    for i in range(n):                                          # 遍历所有特征
        rangeMin = dataMatrix[:,i].min()                        # 找到特征中最小的值
        rangeMax = dataMatrix[:,i].max()                        # 和最大值
        stepSize = (rangeMax - rangeMin) / numSteps             # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:                        # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)    # 阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)   # 预测类别
                # print(predictedVals)
                errArr = mat(ones((m, 1)))                      # 误差矩阵初始化为全1，代表预测值不等于实际值
                errArr[predictedVals == labelMat] = 0           # 如果预测类别等于实际类别，对应的误差应为0
                weightedError = D.T * errArr                    # 将误差向量和权重向量进行内积,得到权值误差
                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the wightedError is: %.3f' %\
                #       (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # print(bestStump, minError, bestClasEst)
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    '''
    使用Adaboost算法提升弱分类器性能
    :param dataArr:           数据矩阵
    :param classLabels:       数据标签
    :param numIt:             最大迭代次数
    :return:  weakClassArr -  单层决策树数组，训练好的分类器
              aggClassEst  -  类别估计累计值
    '''
    weakClassArr = []
    m = shape(dataArr)[0]                                        # 数据集的个数
    D = mat(ones((m,1))/m)                                       # 初始化权重，D为概率分布向量，其所有元素之和等于1
    aggClassEst = mat(zeros((m, 1)))                             # 每个数据点的类别估计累计值
    for i in range(numIt):                                       # 循环迭代numIt次，或者训练误差为0时，停止for循环
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)   # 构建具有最小错误率的单层决策树，最小错误率，类别估计向量
        # print('D:', D.T)                                                   # 输出数据权重，便于理解循环迭代过程
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))          # 分类器的权重alpha，基于每个弱分类器的错误率而得
                                                                           # max(error, 1e-16)确保在没有错误时不会发生除零溢出
        bestStump['alpha'] = alpha                                         # 将分类器的权重alpha加入到字典bestStump中
        weakClassArr.append(bestStump)                                     # 将每次所得字典添加到列表中
        # print('classEst:', classEst.T)                                     # 输出类别估计向量
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)        # exp(-alpha)
        D = multiply(D, exp(expon))                                        # Di * exp(-alpha)
        D = D / D.sum()                                                    # 更新权重D
        aggClassEst += alpha * classEst                                    # 计算类别估计累计值
        # print('aggClassEst:', aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))  # 计算训练误差
        errorRate = aggErrors.sum()/m                                      # 计算训练错误率
        print('total Error: ', errorRate, '\n')                            # 输出训练错误率
        if errorRate == 0.0:                                               # 如果训练误差为零，停止迭代循环训练
            break
    return weakClassArr, aggClassEst

def adaClassify(dataToClass, classifierArr):
    '''
    利用训练出的多个弱分类器进行分类的函数
    :param dataToClass:      待分类的数据
    :param classifierArr:    包含所有弱分类器的字典
    :return:                 待分类数据的类别标签
    '''
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))                                     # 初始化类别估计累计值
    for i in range(len(classifierArr)):                                  # 遍历所有弱分类器字典
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                classifierArr[i]['thresh'], classifierArr[i]['ineq'])    # 在第i个弱分类器下对数据进行估计，得出类别标签
        aggClassEst += classifierArr[i]['alpha'] * classEst              # 通过与第i个弱分类器中的权重值alpha进行乘积，得出
                                                                         # 类别估计累计值
        # print(aggClassEst)
    return sign(aggClassEst)                                             # 返回在多个弱分类器的估计下，得出的估计结果

def loadDataSet(fileName):
    '''
    加载数据
    :param fileName:      文件名
    :return:  dataMat   - 数据矩阵
              labelMat  - 标签矩阵
    '''
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    '''
    绘制ROC曲线，并输出ROC曲线面积
    :param predStrengths: 分类器的预测强度
    :param classLabels:   类别标签
    :return:              ROC曲线
    '''
    cur = (1.0, 1.0)                                # 画笔位置,从(1.0, 1.0)开始画图，一直到(0.0, 0.0)
    ySum = 0.0                                      # 计算ROC
    numPosClas = sum(array(classLabels) == 1.0)     # 计算正例数目
    yStep = 1/float(numPosClas)                     # y轴步进数目
    xStep = 1/float(len(classLabels) - numPosClas)  # x轴步进数目
    sortedIndicies = predStrengths.argsort()        # 将预测强度按照由小到大排序，得到排序索引

    # matplotlib.rcParams['font.family'] = 'STSong'
    # matplotlib.rcParams['font.size'] = 14

    fig = plt.figure()                              # 画图
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:        # 将排序好的预测强度数组或矩阵转换成列表
        if classLabels[index] == 1.0:               # 如果标签为1，沿y轴方向下降一个步长
            delX = 0
            delY = yStep
        else:                                       # 对于其它类别的标签，沿x轴方向下降一个步长
            delX = xStep
            delY = 0
            ySum += cur[1]                          # 矩形高度的和随着在x轴方向上的移动而渐次增加
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c= 'r')  # 画出每次移动后的位置
        cur = (cur[0] - delX, cur[1] - delY)                               # 更新画笔位置点
    ax.plot([0, 1], [0, 1], 'b--')                                         #
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve for Adaboost Horse Colic Detection System')
    plt.xlabel('真阳率', fontproperties= 'STSong',fontsize= 12)
    plt.ylabel('假阳率', fontproperties= 'STSong', fontsize= 12)
    plt.title('AdaBoost马疝病检测系统的ROC曲线', fontproperties= 'STSong', fontsize= 12)
    ax.axis([0, 1, 0, 1])
    print('The area Under the Curve is: ', ySum * xStep)
    plt.show()


def main():
    dataMat, classLabels = loadSimpData()
    # showData(dataMat, classLabels)
    D = mat(ones((5, 1))/5)
    # buildStump(dataMat, classLabels, D)
    # classifierArr = adaBoostTrainDS(dataMat, classLabels, 9)
    # print(classifierArr)
    # labels = adaClassify([[5, 5],[0, 0]], classifierArr)
    # print(labels)
    path1 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\AdaBoost\horseColicTraining2.txt'
    path2 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\AdaBoost\horseColicTest2.txt'

    # **********在分类器数目为10时，训练错误率和测试错误率**************************************************
    # datArr, labelArr = loadDataSet(path1)
    # classifierArr = adaBoostTrainDS(datArr, labelArr, 10)
    # for i in range(len(classifierArr)):
    #     print(classifierArr[i])

    # testArr, testLabelArr = loadDataSet(path2)
    # prediction10 = adaClassify(testArr, classifierArr)
    # print('To get the number of misclassified examples type in:')
    # errArr = mat(ones((67, 1)))
    # errNum = errArr[prediction10 != mat(testLabelArr).T].sum()
    # print('the error number is: %d' % errNum)
    # errRate = errArr[prediction10 != mat(testLabelArr).T].sum()/67
    # print('the error rate is: %.2f%%' % (errRate * 100))
    # print('the error rate is: {:.2%}'.format(errRate))
    # *************************************************************************************************

    # **********采用Adaboost算法对马疝病检测的训练集和测试集，在不同分类器数目下的训练错误率和训练错误率*********
    # datArr, labelArr = loadDataSet(path1)
    # testArr, testLabelArr = loadDataSet(path2)
    # m1 = shape(datArr)[0]
    # m2 = shape(testArr)[0]
    # ParaList = [1, 10, 50, 100, 500, 1000, 10000]
    # OutPara = []
    # for i in ParaList:
    #     classifierArr = adaBoostTrainDS(datArr, labelArr, i)
    #     prediction1 = adaClassify(datArr, classifierArr)
    #     errArr1 = mat(ones((m1, 1)))
    #     errRate1 = errArr1[prediction1 != mat(labelArr).T].sum()/m1
    #
    #     prediction2 = adaClassify(testArr, classifierArr)
    #     errArr2 = mat(ones((m2, 1)))
    #     errRate2 = errArr2[prediction2 != mat(testLabelArr).T].sum()/m2
    #     # Para = ['{:>5d}'.format(i), '{:.2%}'.format(errRate1), '{:.2%}'.format(errRate2)]
    #     Para = [i, errRate1, errRate2]
    #     OutPara.append(Para)
    # Results = mat(OutPara)
    # print('分类器数目  训练错误率(%) 测试错误率(%)')
    # print(Results)
    # ***************************************************************************************************

    # **************************绘制ROC曲线***************************************************************
    dataArr, labelArr = loadDataSet(path1)
    classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
    plotROC(aggClassEst.T, labelArr)
    # *******************************************************************************************

if __name__ == '__main__':
    main()
