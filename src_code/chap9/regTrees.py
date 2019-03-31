#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/23 9:34
# @Author  : Despicable Me
# @Email   : 
# @File    : regTrees.py
# @Software: PyCharm
# @Explain :
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    '''
    函数解释：加载数据集
    :param fileName:   文件名
    :return:           数据集
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))       # 将每行映射为浮点数
        dataMat.append(fltLine)
    return dataMat

def DataShow(dataMat):
    m, n = shape(dataMat)
    xArr = dataMat[:,0:n-1].copy()
    yArr = dataMat[:,-1].copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArr.A, yArr.A, c='g', s=10)
    # plt.axis([-0.2,1.2,-1.0,2.0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def Regre_Show(dataMat, w1, w2, Val):
    '''
    函数说明：回归曲线
    :param dataMat:   数据集
    :param w1:        树模型1的回归系数
    :param w2:        树模型2的回归系数
    :param Val:       分界值
    :return:
    '''
    m, n =shape(dataMat)
    yHat = array(zeros((m,1)))
    xArr = dataMat[:,0:n-1].copy().A
    yArr = dataMat[:,-1].copy().A
    Index = argsort(xArr,0)            # 将数据xArr按0轴升序排列后的索引值
    xArr1 = xArr[Index].copy()

    for i in range(m):
        if xArr1[i] <= Val:
            yHat[i,:] = w2[0] + w2[1] * xArr1[i]
        else:
            yHat[i,:] = w1[0] + w1[1] * xArr1[i]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArr, yArr, c='g', s=10)
    ax.plot(xArr1.flatten(), yHat.flatten(), c='r')
    plt.show()


def binSplitDataSet(dataSet, feature, value):
    '''
    函数说明：通过数组过滤方式将数据集切分成两个子集并返回
    :param dataSet:  待切分数据集
    :param feature:  特征
    :param value:    特征(某个)值
    :return:         两个数据子集
    '''
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1

def regLeaf(dataSet):
    '''
    函数说明：函数确定不再对数据进行切分时，生成叶子节点
    :param dataSet:   数据集
    :return:          叶子节点模型
    '''
    return mean(dataSet[:,-1])     # 叶子节点模型实为目标变量的均值

def regErr(dataSet):
    '''
    函数说明： 误差估计函数，在给定数据集上计算目标变量的平方误差：方差 * n
    :param dataSet:      数据集
    :return:             目标变量的总方差
    '''
    return var(dataSet[:,-1]) * shape(dataSet)[0]   # 方差 * 样本总数

def chooseBestSplit(dataSet, leafType= regLeaf, errType=regErr, ops=(1,4)):
    '''
    函数说明：回归树的切分函数，找到最佳二元切分方式
    :param dataSet:     数据集
    :param leafType:    叶子节点生成函数
    :param errType:     误差估计函数
    :param ops:
    :return:            如果找到一个切分方式，则返回特征编号和切分特征值
    '''
    tolS = ops[0]             # 为容许的误差下降值
    tolN = ops[1]             # 切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:   # 如果所有特征值相等，则不需要再切分
        return None, leafType(dataSet)
    m, n =shape(dataSet)
    S = errType(dataSet)                             # 计算当前数据集的误差，用于与新切分误差进行对比，
                                                     # 来检查新切分能否降低误差
    bestS = inf                                      # 初始化最佳误差值
    bestIndex = 0                                    # 初始化最佳特征索引
    bestValue = 0                                    # 初始化最佳特征值
    for featIndex in range(n - 1):                   # 遍历 n -1个特征
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):   # 遍历每个特征中出现的所有特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)     # 计算新切分误差
            if newS < bestS:                         # 如果新切分误差小于最佳误差，更新最佳误差
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S -  bestS) < tolS:                          # 如果误差减小不大，则退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):     # 如果切分出的数据集小于用户定义的参数tolN则退出
        return None, leafType(dataSet)
    return bestIndex, bestValue                                # 如果所有提前终止条件都不满足，返回切分特征和特征值

def createTree(dataSet, leafType= regLeaf, errType= regErr, ops=(1,4)):
    '''
    函数说明：树构建函数
    :param dataSet:   数据集
    :param leafType:  建立叶节点的函数
    :param errType:   误差计算函数
    :param ops:       树构建所需其他参数的元组
    :return:
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:                     # 满足停止条件时返回叶节点值
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):                           # 判断对象是否为叶子节点
    return (type(obj).__name__ == 'dict')  # 不为叶子节点返回1

def getMean(tree):
    '''
    函数说明： 从上往下遍历树知道叶子节点为止，如
    果找到两个叶子节点，返回它们的平均值
    :param tree:    待遍历的树
    :return:        树平均值
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree, testData):
    '''
    函数说明：对tree进行(后)剪枝
    :param tree:       待剪枝的树
    :param testData:   剪枝所需的测试数据
    :return:
    '''
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):                        # #
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])   # #
    if isTree(tree['left']):                                                   # # 检查分支是子树还是节点，
        tree['left'] = prune(tree['left'], lSet)                               # # 如果是子树，调用prune函数对该子树进行剪枝
    if isTree(tree['right']):                                                  # #
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):                 # 剪枝完成后还需检查它们是否还是子树，若果不是
                                                                               # 就进行合并
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) +\
                       sum(power(rSet[:,-1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMean = (tree['left'] + tree['right'])/2.0
        if errorMean < errorNoMerge:                                           # 对合并前后的误差作对比，如果合并误差比不合并
            print('mergeing')                                                  # 误差小，就合并，并返回合并后的树
            return treeMean
        else:
            return tree
    else:
        return tree
def linearSolve(dataSet):
    '''
    函数说明： 线性模型求解参数，将数据集格式化成目标变量X,Y
    :param dataSet:   数据集
    :return:     ws - 回归系数
                 X  - 输入数据
                 Y  - 输出数据
    '''
    m, n = shape(dataSet)
    X = mat(ones((m, n )))
    Y = mat(ones((m, 1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n\
                        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    '''
    函数说明：生成叶子节点模型
    :param dataSet:   数据集
    :return:      ws- 回归系数
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    '''
    函数说明：在给定的数据集计算预测误差
    :param dataSet:
    :return:
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))

def regTreeEval(model, inDat):
    '''
    函数说明：对回归树叶节点进行预测
    :param model: 回归树模型
    :param inDat: 输入数据
    :return:      返回预测值
    '''
    return float(model)

def modelTreeEval(model, inDat):
    '''
    函数说明：对模型树节点进行预测
    :param model:   模型树
    :param inDat:   输入数据
    :return:        返回预测值
    '''
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:,1:n+1] = inDat
    return float(X * model)

def treeForceCast(tree, inData, modelEval= regTreeEval):
    '''
    函数说明：自顶向下遍历整棵树，直到命中叶子节点为止
    :param tree:      给定树结构
    :param inData:    输入数据
    :param modelEval: 对叶节点数据进行预测的函数的引用
    :return:          返回预测值
    '''
    if not isTree(tree):                 # 若为叶子节点，返回预测值
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForceCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForceCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForceCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i,0] = treeForceCast(tree, mat(testData[i]), modelEval)
    return yHat

# def main():
#     path1 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression Trees\ex00.txt'
#     path2 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression Trees\ex0.txt'
#     path3 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression Trees\ex2.txt'
#     path4 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression Trees\ex2test.txt'
#     path5 = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression Trees\exp2.txt'
#     path6 = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\9.RegTrees\\bikeSpeedVsIq_train.txt'
#     path7 = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\9.RegTrees\\bikeSpeedVsIq_test.txt'
#
#     # # # *********************** 数据集ex00 ***********************************
#     # myData1 = loadDataSet(path1)
#     # myMat1 = mat(myData1)
#     # # DataShow(myMat1)
#     # retTree = createTree(myMat1, ops=(0,1))
#     # print(retTree)
#
#     # # *********************** 数据集ex0  ************************************
#     # myData2 = loadDataSet(path2)
#     # myMat2 = mat(myData2)
#     # myMat3= myMat2[:,1:3].copy()
#     # DataShow(myMat3)
#     # retTree = createTree(myMat3)
#     # print(retTree)
#
#     # # # *********************** 剪枝操作 ***************************************
#     # myDat3 = loadDataSet(path3)
#     # myMat3 = mat(myDat3)
#     # myTree = createTree(myMat3, ops= (0,1))
#     # myDatTest = loadDataSet(path4)
#     # myMat3Test = mat(myDatTest)
#     # retTree = prune(myTree, myMat3Test)
#     # print(retTree)
#     # # # **********************************************************************
#
#     # # # *********************** 模型树 ****************************************
#     # myDat4 = loadDataSet(path5)
#     # myMat4 = mat(myDat4)
#     # # DataShow(myMat4)
#     # retTree = createTree(myMat4, modelLeaf, modelErr, ops=(1, 10))
#     # Middle_Val = retTree['spVal']                      # 分界值
#     # ws1 = retTree['left'].A                            # 回归系数1
#     # ws2 = retTree['right'].A                           # 回归系数2
#     # print(retTree)
#     # print(ws1.flatten(),'\n', ws2.flatten())
#     # Regre_Show(myMat4, ws1, ws2, Middle_Val)           # 展示回归效果
#     # # # ***********************************************************************
#
#
#     # 》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
#     # # ****************************** 树回归与线性回归的比较 *********************
#     myDat5 = loadDataSet(path6)
#     myDat6 = loadDataSet(path7)
#     trainMat = mat(myDat5)
#     testMat = mat(myDat6)
#     # DataShow(trainMat)
#
#     # # ****************************** 回归树模型 *************************
#     myTree1 = createTree(trainMat, ops=(1,20))
#     yHat1 = createForceCast(myTree1, testMat[:,0])
#     corr1 =corrcoef(yHat1, testMat[:,1], rowvar= 0)[0,1]
#
#     # # ****************************** 模型树模型 **************************
#     myTree2 = createTree(trainMat, modelLeaf, modelErr, (1,20))
#     yHat2 = createForceCast(myTree2, testMat[:,0], modelTreeEval)
#     corr2 = corrcoef(yHat2, testMat[:,1], rowvar= 0)[0,1]
#
#     # # ****************************** 标准线性模型 ************************
#     ws, X, Y = linearSolve(trainMat)
#     m = shape(trainMat)[0]
#     yHat3 = mat(ones((m,1)))
#     for i in range(m):
#         yHat3[i] = testMat[i,0] * ws[1,0] + ws[0,0]
#     corr3 = corrcoef(yHat3, testMat[:,1], rowvar= 0)[0,1]
#
#     # # ***************** 输出相关系数比较结果 *******************************
#     print('三种回归模型下的相关系数比较：(相关系数越大，模型越好)')
#     print('回归树模型：%f' % corr1, '\n' '模型树模型：%f' % corr2, '\n' '线性回归模型：%f' % corr3)
#
#     # # ***************** 三种回归方法下所得出的预测结果可视化比较 ***************
#     xdata = testMat[:,0].copy().A         # 将输入数据拷贝到xdata,并转化为数组形式
#     ydata = testMat[:,1].copy().A
#
#     X_index = argsort(xdata,0).copy()     # 输入数据升序排列的索引
#     X = xdata[X_index,0].copy()           # 输入数据升序排列后的数据存入X
#
#     # # 可视化展示
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(xdata, ydata, c='g', s= 9, label= 'Original')                  # 原始数据
#     ax.plot(X, yHat1[X_index,0].A, c='b', linewidth=3, label= 'RegreTree')    # 回归树预测值
#     ax.plot(X, yHat2[X_index,0].A, c='r', linewidth=3, label= 'ModelTree')    # 模型树预测值
#     ax.plot(X, yHat3[X_index,0].A, c='y', linewidth=3, label= 'LinearRegre')  # 线性回归预测值
#     plt.legend(loc='upper left')
#     plt.show()
#     # # ********************************************************************
#     # 》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》
#
#
#
# if __name__ == '__main__':
#     main()