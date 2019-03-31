#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/21 11:41
# @Author  : Despicable Me
# @Email   :
# @File    : ridge_regreesion.py
# @Software: PyCharm
# @Explain :

from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from time import sleep
import json
import urllib.request
from bs4 import BeautifulSoup

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

def ridgeRegres(xMat, yMat, lam = 0.2):
    '''
    计算岭回归系数
    :param xMat:    输入矩阵
    :param yMat:    输出矩阵
    :param lam:     lambda的值，默认为0.2
    :return:   ws - 回归系数
    '''
    xTx = xMat.T * xMat                      # X.T * X
    denom = xTx + lam * eye(shape(xMat)[1])  # X.T * X + lambda * I
    if linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    '''
    在一组lambda上测试，得到不同的回归系数
    :param xArr:     输入矩阵
    :param yArr:     输出矩阵
    :return: WMat -  回归系数组成的矩阵
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)                          # 求输出矩阵的平均值，按列(axis = 0)
    yMat = yMat - yMean                            # 输出标准化
    xMeans = mean(xMat, 0)                         # 求输入矩阵的平均值，按列(axis = 0),每个特征值的均值
    xVar = var(xMat, 0)                            # 求输入矩阵的方差值，按列(axis = 0),每个特征值的方差
    xMat = (xMat - xMeans) / xVar                  # 输入标准化处理
    numTestPts = 30                                # lambda个数
    wMat = zeros((numTestPts, shape(xMat)[1]))     # 初始化存储ws的矩阵
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))  # 求不同lambda值下的回归系数
        wMat[i,:] = ws.T
    return wMat                                    # 返回回归系数矩阵

def regularize(xArr):
    '''
    输入矩阵标准化处理
    :param xArr:    待处理数据
    :return:        标准化之后数据
    '''
    xMat = xArr.copy()
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    return xMat

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

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    '''
    前向逐步线性回归
    :param xArr:      输入数据列表
    :param yArr:      输出数据列表
    :param eps:       每次迭代需要调整的步长，默认步长为0.01
    :param numIt:     最大迭代次数，默认最大迭代数为100
    :return:          返回回归系数的列表矩阵
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMeans = mean(yMat, 0)
    yMat = yMat - yMeans
    xMat = regularize(xMat)                         # 输入数据标准化
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))                   # 回归系数初始化
    ws = zeros((n, 1))                              # 创建ws保存参数w的值
    wsTest = ws.copy()                              # 副本ws
    wsMax = ws.copy()                               # 最大ws的初始化
    for i in range(numIt):
        print(ws.T)                                 # 每次迭代打印出参数w用于分析算法执行的过程和效果
        lowestError = inf                           # 每次迭代设置最小误差为无穷
        for j in range(n):                          # 计算每个特征对应的参数w
            for sign in [-1, 1]:                    # 每次计算增加或减少该特征对误差的影响
                wsTest = ws.copy()
                wsTest[j] += eps * sign             # 更新第j个特征对应的参数w
                yTest = xMat * wsTest               # 参数每次改变都计算预测输出
                rssE = rssError(yMat.A, yTest.A)    # 计算预测误差
                if rssE < lowestError:              # 对比所计算预测误差与给定误差的大小
                    lowestError = rssE              # 取最小误差
                    wsMax = wsTest                  # 取最小误差对应的参数w
        ws = wsMax.copy()                           #
        returnMat[i,:] = ws.T                       # 将每次计算所得最优参数w保存至参数矩阵
    return returnMat                                # 返回参数矩阵

'''
由于谷歌api已关闭，采用下面的代码实现对乐高积木数据的爬取
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):

    调用Google购物API并保证数据抽取的正确性
    :param retX:      包含各个特征的列表
    :param retY:      价格表
    :param setNum:    循环迭代次数
    :param yr:        年份
    :param numPce:    乐高部件数
    :param origPrc:   原始价格
    :return:

    sleep(10)
    myAPIstr = 'get from code.google.com'
    searchURL = 'https://www.googleapis.com/shapping/seaarch/v1/public/products?\
    key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print ('%d\t%d\t%d\t%f\t%f' % (yr, numPce, newFlag, origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc, sellingPrice])
                    retY.append(sellingPrice)
        except: print('problem with item %d' % i)
'''

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    '''
    从页面读取数据，生成retX,retY列表
    :param retX:       数据X(输入数据)
    :param retY:       数据Y（输出数据，价格)
    :param inFile:     HTML文件
    :param yr:         年份
    :param numPce:     乐高部件数目
    :param origPrc:    原价
    :return:           无
    '''
    # ***** 打开并读取HTML文件 ******
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")

    i = 1
    # 根据html页面结构进行解析
    currentRow = soup.find_all('table', r = '%d' % i)

    while(len(currentRow) != 0):
        currentRow = soup.find_all('table', r = '%d' % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新的标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):       # 检查字符串'new'或者'nisd'是否包含在字符串lwrTitle中，如果有，返回索引值，没有返回-1
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已标志出售，我们只收集已出售的数据
        soldUnicode = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicode) == 0:
            print('商品 # %d  没有出售' % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套件价格
            if sellingPrice > origPrc * 0.5:
                print('%d\t%d\t%d\t%f\t%f' % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r= '%d' % i)


def setDataCollect(retX, retY):
    '''
    依次读取六种乐高套装的数据，并生成数据矩阵
    :param retX:    输入空列表
    :param retY:    输出空列表
    :return:        包含六种乐高套装数据的矩阵
    '''
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)

def crossValidation(xArr, yArr, numVal=10):
    '''
    函数说明：交叉验证测试岭回归
    :param xArr:    原始输入数据
    :param yArr:    原始输出数据
    :param numVal:  指定交叉验证次数，默认为10折交叉验证
    :return:
    '''
    m = len(yArr)
    indexList = list(range(m))              # 创建索引列表
    errorMat = zeros((numVal, 30))    # 初始化误差矩阵
    for i in range(numVal):                # 十次循环
        trainX = []
        trainY = []
        testX = []
        testY = []
        random.shuffle(indexList)     # 将序列indexList随机排序
        for j in range(m):            # 遍历所有数据,打乱次序
            if j < m * 0.9:           # 划分数据集:90%训练集，10%测试集
                trainX.append(xArr[j])
                trainY.append(yArr[j])
            else:
                testX.append(xArr[j])
                testY.append(yArr[j])
        wMat = ridgeTest(trainX, trainY)  # 获得30个不同lambda下的岭回归系数
        for k in range(30):               # 遍历所有的岭回归系数
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain)/varTrain
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)    # 根据ws预测y值
            errorMat[i,k] = rssError(yEst.T.A, array(testY))     # 统计误差，10折验证中每个lambda对应的误差
    meanErrors = mean(errorMat,0)            # 计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))         # 找到平方误差中的最小值
    bestWeights = wMat[nonzero(meanErrors== minMean)]   # 平方误差最小的lambda对应的回归系数为最佳回归系数
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights/varX                 # 数据经过标准化，因此需要还原
    # print('The best model from Ridge Regression is:\n', unReg)
    const = -1 * sum(multiply(meanX, unReg)) + mean(yMat)
    ws = [const, unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]]
    print('(岭回归下得出的)价格决定公式: %f%+f*年份%+f*部件数%+f*是否为新%+f*原价' % (const, unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))
    return ws

def main():
    path = 'D:\Projects_Python\Dong\GitHub Files\Machine-Learning\Regression\\abalone.txt'
    abX, abY = loadDataSet(path)

    # # ******************************* 绘制岭回归系数随log(lambda)的变化 ***********************
    # ridgeWeights = ridgeTest(abX, abY)                     # 计算岭回归系数矩阵
    # print('The regression coefficient of ridge is:')
    # print(ridgeWeights)
    # ***********************************
    # ***********************************
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(arange(-10,20,1).T, ridgeWeights)
    # ax.tick_params(direction= 'in')                 # 调整坐标轴刻度的朝向，in=朝内，out=朝外，inout=内外均有
    # plt.xlabel('log(lambda)')
    # plt.axis([-10, 20, -1.0, 2.5])
    # plt.show()
    # # ***************************************************************************************

    # # ******************************* 逐步线性回归算法 ******************************************
    # ridge_ws = stageWise(abX, abY, 0.005, 1000)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridge_ws)
    # ax.tick_params(direction= 'in')                 # 调整坐标轴刻度的朝向，in=朝内，out=朝外，inout=内外均有
    # # plt.xlabel('log(lambda)')
    # plt.axis([0, 1000, -1.0, 1.5])
    # plt.show()
    # # *****************************************************************************************

    # ******************************* 乐高积木价格预测 ******************************************
    lgX = []           # 输入空列表
    lgY = []           # 输出空列表
    setDataCollect(lgX, lgY)
    m, n = shape(lgX)
    print(m, n)
    lgX1 = mat(ones((m, n + 1)))
    lgX1[:, 1:n + 1] = mat(lgX)
    # # *********************** 标准线性回归 ***********************
    ws1 = standRegres(lgX1, lgY).T
    print('(标准线性回归下得出的)价格决定公式: %f%+f*年份%+f*部件数%+f*是否为新%+f*原价' % (ws1[0,0], ws1[0,1], ws1[0,2], ws1[0,3], ws1[0,4]))
    price_standard = lgX1 * ws1.T       # 标准线性回归预测结果
    # # *********************** 岭回归 ****************************
    ws2 = crossValidation(lgX, lgY, 10)
    # # *********************** 两种回归方法所得预测结果和实际结果对比******
    price_ridge = lgX1 * mat(ws2).T
    plt.figure()
    plt.plot(mat(lgY).T, c='b', label='Real')
    plt.plot(price_standard, c='g', label='standard')
    plt.plot(price_ridge, c='r', label='ridge')
    plt.legend(loc = 'upper left')
    plt.show()
    # # ***************************************************************

if __name__ == '__main__':
    main()
