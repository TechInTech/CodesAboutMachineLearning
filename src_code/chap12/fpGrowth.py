#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/1/1 22:03
# @Author  : Despicable Me
# @Email   : 
# @File    : fpGrowth.py
# @Software: PyCharm
# @Explain :
class treeNode:
    '''
    FP树的类定义
    '''

    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):
    '''
    函数说明：FP树构建函数，构建FP树过程中，会遍历数据集两次
    :param dataSet:    数据集
    :param minSup:     最小支持度
    :return:
    '''
    headerTable = {}  # 头指针表
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):  # 扫描头指针表
        if headerTable[k] < minSup:  # 如果频繁项出现次数少于指定最小支持度，删除当前项
            del (headerTable[k])
    freqItemSet = set(headerTable.keys())  # 创建过滤后的项集的不变集合
    if len(freqItemSet) == 0:  # 如果所有项都不频繁，不需要进一步处理
        return None, None
    for k in headerTable:  # 遍历头指针列表
        # 格式化： dist{元素key: [元素次数, None]}
        headerTable[k] = [headerTable[k], None]  # 对指针表进行扩展

    # # 创建FP树
    retTree = treeNode('Null Set', 1, None)  # 创建FP树中只包含空集合的根节点
    for tranSet, count in dataSet.items():  #
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1:], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    """
    父节点非空，将记录当前节点
    :param leafNode:     查询的节点对于的nodeTree
    :param prefixPath:   要查询的节点值
    :return:
    """
    if leafNode.parent is not None:  # 父节点非空，将记录当前节点
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    """
    发现以给定元素项结尾的所有路径，
    为给定元素项生成一个条件模式基
    :param basePat:      要查询的节点值
    :param treeNode:     查询的节点所在的当前nodeTree
    :return:  con_pats - 对非basePat的倒叙值作为key,赋值为count数
    """
    cond_pats = {}
    # 对 treeNode的link进行循环
    while treeNode is not None:
        prefixPath = []
        # 寻找改节点的父节点，相当于找到了该节点的频繁项集
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            # 对非basePat的倒叙值作为key,赋值为count数
            # prefixPath[1:] 变frozenset后，字母就变无序了
            # cond_pats[frozenset(prefixPath)] = treeNode.count
            cond_pats[frozenset(prefixPath[1:])] = treeNode.count
        #  递归，寻找改节点的下一个 相同值的链接节点
        treeNode = treeNode.nodeLink
    return cond_pats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    '''递归查找频繁项集的函数'''
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        # 挖掘条件 FP-tree, 如果myHead不为空，表示满足minSup {所有的元素+(value, treeNode)}
        if myHead is not None:
            print('conditional tree for:', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    '''
    函数说明：创建初始化字典
    :param dataSet:   数据集
    :return:          键值为1的字典
    '''
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) not in retDict.keys():
            retDict[frozenset(trans)] = 1
        else:
            retDict[frozenset(trans)] += 1
    return retDict


def main():
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    # print(initSet)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()
    for item in myHeaderTab.items():
        print(item)


    # condP = findPrefixPath('x', myHeaderTab['x'][1])
    # # print(condP)
    # freqItems = []
    # mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    # print("freqItemList: \n", freqItems)


    # # # **************************** 新闻网站点击流中挖掘 ***********************
    # path = 'D:\Projects_Python\Dong\GitHub Files\AiLearning\db\\12.FPGrowth\kosarak.dat'
    # parsedDat = [line.split() for line in open(path).readlines()]     # 导入数据
    # initSet = createInitSet(parsedDat)                                # 初始集格式化
    # myFPtree, myHeaderTab = createTree(initSet, 100000)               # 构建FP树
    # myFreqList = []                                                   # 创建空列表保存频繁项集
    # mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)      #
    # print(len(myFreqList), '\n', myFreqList)


if __name__ == '__main__':
    main()
