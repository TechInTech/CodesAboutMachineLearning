#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/26 9:42
# @Author  : Despicable Me
# @Email   : 
# @File    : kMeans.py
# @Software: PyCharm
# @Explain : K-均值聚类算法
from numpy import *
import matplotlib.pyplot as plt
import json
import urllib.request
from time import sleep
import urllib.parse
import matplotlib

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    '''
    函数说明： 计算两个向量间的欧氏距离
    :param vecA:  向量A
    :param vecB:  向量B
    :return:      欧氏距离
    '''
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    '''
    函数说明：为给定数据集构建一个包含k个随机质心的集合
    :param dataSet:   数据集
    :param k:         质心数
    :return:          包含k个质心的集合
    '''
    n = shape(dataSet)[1]               # 数据集的列数
    centroids = mat(zeros((k, n)))      # 初始化质心矩阵
    for j in range(n):                  # 遍历数据集的每列
        minJ = min(dataSet[:,j])        # 每列的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)            # 最小最大值的间隔
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)  # 保证质心在整个数据集的边界内
    return centroids

def kMeans(dataSet, k, distMeas= distEclud, createCent= randCent):
    '''
    函数说明：
    :param dataSet:      数据集
    :param k:            聚类个数
    :param distMeas:     距离计算方式
    :param createCent:   随机质心产生函数
    :return: centroids - 质心位置
        clusterAssment -
    '''
    m, n = shape(dataSet)
    clusterAssment = mat(zeros((m,2)))      # 簇分配结果矩阵，1列记录簇索引值，1列存储误差
    centroids = createCent(dataSet, k)      # 创建随机质心
    clusterChanged = True                   # 标识变量
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1    # 初始化最小距离及索引位置(类标签)
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])   #寻找最近质心
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist ** 2      # 簇分配结果，保存最小索引和最小距离，最小索引值为对应的数据集的某行的类标签
        # print(centroids)
        for cent in range(k):                                               # 更新质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]] # 通过数组过滤获得给定簇的所有点
            centroids[cent,:] = mean(ptsInClust, axis=0)                    # 计算所点的均值
    return centroids, clusterAssment                    # 返回所有的类质心和点分配结果

def biKmeans(dataSet, k, distMeas= distEclud):
    '''
    函数说明：二分K-均值聚类算法
    :param dataSet:   数据集
    :param k:         簇数目
    :param distMeas:  距离计算方式
    :return:
    '''
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids0 = mean(dataSet, axis=0).tolist()[0]    # 计算整个数据集的质心,并使用一个列表来保留所有的质心
    cenList = [centroids0]                            # 簇列表
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroids0), dataSet[j,:])**2   # 计算质心到数据集每个点的误差值
    while (len(cenList) < k):                         # 若不满足指定簇数目，继续循环
        lowestSSE = inf                               # 初始化最小SSE为inf,用于比较划分前后的SSE
        # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
        for i in range(len(cenList)):                 # 遍历所有的簇，决定最佳的簇进行划分
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]  # 将数据集中第i簇看作是一个子数据集
            # 将ptsInCurrCluster输入到函数kMeans中进行处理,k=2,
            # kMeans会生成两个质心(簇),同时给出每个簇的误差值
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)    # 生成两个质心簇，同时给出每个簇的误差

            sseSplit = sum(splitClustAss[:,1])                                    # 计算第i簇的误差之和
            sseNOTSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])  # 计算非第i簇(剩余数据集)的误差之和

            # 将 sseSplit + sseNOTSplit看做是本次划分的误差
            print('sseSplit, and notSplit:', sseSplit, sseNOTSplit)

            # 如果划分的SSE值最小，保留此次划分
            if (sseSplit + sseNOTSplit) < lowestSSE:
                bestCentToSplit = i                     #
                bestNewCents = centroidMat              # 保留划分的质心簇
                bestClustAss = splitClustAss.copy()     # 保留最佳划分误差
                lowestSSE = sseSplit + sseNOTSplit      # i簇与非i簇的误差之和
        # 找出最好的簇分配结果
        # 调用kmeans函数并且指定簇数为2时,会得到两个编号分别为0和1的结果簇
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(cenList)

        # 更新为最佳质心
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit

        print('the bestCentToSplit is:', bestCentToSplit)
        print('the len of bestClustAss is:', len(bestClustAss))
        # 更新质心列表
        # 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
        cenList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        # 添加bestNewCents的第二个质心
        cenList.append(bestNewCents[1, :].tolist()[0])
        # 重新分配最好簇下的数据(质心)以及SSE
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(cenList), clusterAssment

def geoGrab(stAddress,city):
    apiStem ='http:///where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = urllib.request.urlopen(yahooApi)
    return json.loads(c.read())

def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print('%s\t%f\t%f' % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print('error fetching')
        sleep(1)
    fw.close()

def distSLC(vecA, vecB):
    '''
    函数说明：返回地球表面两点之间的距离
    :param vecA:    点A的坐标(纬度，经度)
    :param vecB:    点B的坐标(纬度，经度)
    :return:
    '''
    a = sin(vecA[0,1] * pi/180) * sin(vecB[0,1] * pi/180)
    b = cos(vecA[0,1] * pi/180) * cos(vecB[0,1] * pi/180) *\
        cos(pi * (vecB[0,0] - vecA[0,0])/180)
    return arccos(a + b) * 6371.0

def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])     # 分别对应纬度和经度
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas= distSLC)  # 采用二分k-均值聚类算法

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]                                               # figure的百分比,从figure 10%的位置开始绘制, 宽高是figure的80%
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']       # 标记形状列表
    axprops = dict(xticks= [], yticks= [])

    ax0 = fig.add_axes(rect, label='ax0', **axprops)                          # 在fig上显示两张图
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)                      # ax为背景，ax1绘制在ax0上, frameon= False不显示边框

    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A == i)[0],:]      # 获得第i类在原数据集上的子数据集
        markerStyle = scatterMarkers[i % len(scatterMarkers)]                 # 获得标记符号，当有多个簇时，符号可以循环使用
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
                        ptsInCurrCluster[:,1].flatten().A[0], marker= markerStyle, s= 90)
    ax1.scatter(myCentroids[:,0].flatten().A[0],\
                    myCentroids[:,1].flatten().A[0], marker='+', s= 300)
    plt.show()


def cluster_Show(dataMat, centroids, clusterAss, k):
    '''
    函数说明：聚类结果展示
    :param dataMat:     数据集
    :param centroids:   质心位置
    :param clusterAss:  聚类最小索引值及距离误差平方和
    :param k:           聚类数
    :return:
    '''
    color = 'ygbc'              # 类别颜色字符串
    mark_label = 'spdo'         # 标签类别字符串
    arrow_dict = dict(facecolor= 'red', edgecolor= 'r', shrink=0.1, headlength= 15, headwidth= 10, width=2) # 箭头属性
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(k):
        ax.scatter(dataMat[nonzero(clusterAss[:,0].A == i)[0]].A[:,0],\
                   dataMat[nonzero(clusterAss[:,0].A == i)[0]].A[:,1], c=color[i], marker= mark_label[i], label='class of %d' % i)
        ax.annotate('centroids', xy=(centroids[i,:].A.tolist()[0]), xytext=(0,0), arrowprops= arrow_dict)
    ax.scatter(centroids[:,0].A, centroids[:,1].A, c='black', marker='*', s=50, label='centroids')          # 质点位置

    plt.legend(loc= 'lower left')
    plt.show()

def main():

    # # # ********************** K-均值聚类算法 *******************************
    # path = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\\10.KMeans\\testSet.txt'
    # datMat = mat(loadDataSet(path))
    # mycentroids, clusterAssing = kMeans(datMat, 4)           # 获取质心位置，索引列表
    # cluster_Show(datMat, mycentroids, clusterAssing, 4)      # 绘制聚类结果的散点图，及质心位置
    # # *********************************************************************


    # # ************************ 二分k-均值聚类算法 ****************************
    # path = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\\10.KMeans\\testSet2.txt'
    # datMat = mat(loadDataSet(path))
    # centList, mynewAssment = biKmeans(datMat, 3)
    # print(centList)
    # cluster_Show(datMat, centList, mynewAssment, 3)
    # # **********************************************************************

    # # # ********************** yahoo 地图标记***************************************
    # geoResults = geoGrab('1 VA Center', 'Augusta, ME')
    # print(geoResults)

    # #
    clusterClubs(4)


if __name__ == '__main__':
    main()