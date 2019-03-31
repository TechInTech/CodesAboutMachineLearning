#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/26 21:21
# @Author  : Despicable Me
# @Email   : 
# @File    : apriori.py
# @Software: PyCharm
# @Explain :


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    '''
    函数说明：创建包含所有项的不变集合
    :param dataSet:   数据集
    :return:     C1 - 包含不变集(单元素)的集合
    '''
    C1 = []
    for transaction in dataSet:          # 便利数据集中的每一条记录
        for item in transaction:         # 对每一条记录，遍历记录中的每一项
            if not [item] in C1:         # 如果物品项不在C1中，将其添加进C1
                C1.append([item])        # 注意添加的为物品项的列表
    C1.sort()
    return list(map(frozenset, C1))      # frozenset为‘冰冻’集合，为不可改变集合

def scanD(D, Ck, minSupport):
    '''
    函数说明：
    :param D:            数据集
    :param Ck:           候选项集列表
    :param minSupport:   感兴趣项集的最小支持度
    :return:   retList - 满足最小支持度要求的项集的项集列表L1
           supportData - 最频繁项集的支持度
    '''
    ssCnt = {}           # 空字典
    for tid in D:        # 遍历数据集，tid为单个记录
        for can in Ck:   # 遍历项集
            if can.issubset(tid):                 # 如果项集为记录tid的一部分
                if not ssCnt.__contains__(can):   # 如果项集不在字典ssCnt中，将can添加入字典ssCnt中,并计数
                    ssCnt[can] = 1
                else:                             # 如果已在，数量加1
                    ssCnt[can] += 1
    numItems = float(len(D))                      # 数据集的个数
    retList = []                                  # 创建包含满足最小支持度要求的集合的列表
    supportData = {}                              # 包含支持度的字典
    for key in ssCnt:                             # 遍历字典中每个元素
        support = ssCnt[key]/numItems             # 计算支持度
        if support >= minSupport:                 # 支持度大于最小支持度要求，保存项集
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):    # creates Ck
    '''
    函数说明：输入频繁项集列表 Lk 与返回的元素个数 k，然后输出候选项集 Ck
    :param Lk:    频繁项集列表
    :param k:     返回的项集元素个数
    :return:      元素两两合并的数据集
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:                      # 如果两个集合的前k-2个元素相等，两个集合进行合并
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport= 0.5):
    '''
    函数说明：
    :param dataSet:         数据集
    :param minSupport:      支持度
    :return:                候选项集列表
    '''
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):                 # 当Lk为空时，程序返回L并退出
        Ck = aprioriGen(L[k-2],k)            # 产生不变项集
        Lk, supk = scanD(D, Ck, minSupport)  # 选择满足支持度要求的项集(过滤不满足支持度的项集)
        supportData.update(supk)             # 更新列表(将新列表加入到supportData中)
        L.append(Lk)
        k += 1
    return L, supportData                    # 返回候选集列表L和支持度

def generateRules(L, supportData, minConf= 0.7):
    '''
    函数说明：
    :param L:            频繁项集列表
    :param supportData:  包含那些频繁项集支持数据的字典
    :param minConf:      最小可信度阈值
    :return:             包含可信度的规则列表
    '''
    bigRuleList = []                 # 存放可信度的规则空列表
    for i in range(1, len(L)):
        for freqSet in L[i]:         # 遍历频繁项集
            H1 = [frozenset([item]) for item in freqSet]     # 由频繁项集freqSet创建的只包含单个元素集合的列表
            if(i > 1):                                       # 如果频繁项集元素数目超过2个，考虑对它做进一步合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, br1, minConf= 0.7):
    '''
    函数说明： 对候选规则进行评估
    :param freqSet:      频繁项集
    :param H:            有频繁项集创建的只包含单个元素集合的列表
    :param supportData:  包含那些频繁项集支持数据的字典
    :param br1:          可信度的规则列表
    :param minConf:      最小可信度
    :return:             满足最小可信度要求的规则列表
    '''
    prunedH = []
    for conseq in H:                                                    # 遍历H中的所有项集，并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet - conseq]       # 计算规则的可信度
        if conf >= minConf:                                             # 如果满足最小可信度要求，输出显示，并保存
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH       # 返回满足最小可信度要求的规则列表

def rulesFromConseq(freqSet, H, supportData, br1, minConf= 0.7):
    '''
    函数说明：对规则进行评估
    :param freqSet:      频繁项集
    :param H:            有频繁项集创建的只包含单个元素集合的列表
    :param supportData:  包含那些频繁项集支持数据的字典
    :param br1:          可信度的规则列表
    :param minConf:      最小可信度
    :return:
    '''
    m = len(H[0])                 # 计算H中频繁项集大小
    if (len(freqSet) > (m + 1)):  # 判断
        Hmp1 = aprioriGen(H, m + 1)   # 由项集H生成包含m+1个元素的项集
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)  # 计算满足最小可信度要求的规则列表
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

def main():
    # dataSet = loadDataSet()
    # C1 = createC1(dataSet)
    # # print(list(C1))
    # D = list(map(set, dataSet))
    # print(list(D))

    # # # ********************* scanD *************************************
    # L1, suppData0 = scanD(D, C1, 0.5)
    # print(L1, '\n', suppData0)
    #
    # # ****************************** 利用apriori发现频繁项集 ***************************
    # L, supportData = apriori(dataSet)
    # print(L)
    # print(supportData)
    #
    #
    # # # ******************************
    # # L, supportData = apriori(dataSet, minSupport= 0.5)
    # rules = generateRules(L, supportData, minConf= 0.5)
    # print(rules)

    # # # ******************************************
    # from votesmart import votesmart
    # votesmart.apikey = '49024thereoncewasamanfromnantucket94040'
    # bills = votesmart.votes.getBillsByStateRecent()

    # # ************************* 毒蘑菇 *****************************
    Datapath = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\\11.Apriori\mushroom.dat'
    mushDatSet = [line.split() for line in open(Datapath).readlines()]
    L, supportData = apriori(mushDatSet, minSupport= 0.3)

    for item in L[1]:
        if item.intersection('2'):
            print(item)

    for item in L[3]:
        if item.intersection('2'):
            print(item)


if __name__ == '__main__':
    main()