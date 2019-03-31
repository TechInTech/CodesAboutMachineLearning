#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/26 19:35
# @Author  : Despicable Me
# @Site    : 
# @File    : bayes.py
# @Software: PyCharm
# @Explain :

from numpy import *
import os
import feedparser

def loadDataSet():
    # postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
    #                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
    #                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
    #                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    #                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
    #                ['quit', 'buying', 'worthing', 'dog', 'food', 'stupid']]
    # classVec = [0, 1, 0, 1, 0, 1]                # 1 代表侮辱性的文字， 0代表正常言论
    postingList = [['1', 'S'],
                   ['1', 'M'],
                   ['1', 'M'],
                   ['1', 'S'],
                   ['1', 'S'],
                   ['2', 'S'],
                   ['2', 'M'],
                   ['2', 'M'],
                   ['2', 'L'],
                   ['2', 'L'],
                   ['3', 'L'],
                   ['3', 'M'],
                   ['3', 'M'],
                   ['3', 'L'],
                   ['3', 'L']]
    classVec = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 词集模型
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    # 不同点
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec

# 词袋模型
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1   # 不同点
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)        # P(1)的概率，为二分类问题，则有P(0) = 1 - P(1)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # print(p0Num,'\n', p1Num)
    # print(p0Denom, p1Denom)
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + (1 - log(pClass1))
    # print(p0, p1)
    if  p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):                     # bigString为长字符串，也可以说是整个文本的长字符串
    import re
    listOfTokens = re.split(r'\W*', bigString)     # 正则表达式的分隔符为除单词、数字外的任意字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 文档路径
path1 = 'D:\Projects_Python\DataSets\machinelearninginaction\Ch04\email\spam'
path2 = 'D:\Projects_Python\DataSets\machinelearninginaction\Ch04\email\ham'

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open(path1 + '\%d.txt' % i).read())    # 某一文档的字符列表
        docList.append(wordList)                  # 将所得字符列表按列表形式存放到docListzhong
        fullText.extend(wordList)                 # 将所得字符列表按字符形式存放到fullText中
        classList.append(1)
        wordList = textParse(open(path2 + '\%d.txt' % i, errors= 'ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)           # docList为整个文档（全部邮件），文档每行为一份邮件中的全部字符,
                                                   # 同时将整个文档中出现的字符整理为一个字符向量
    trainingSet = list(range(0,50))
    # ********随机选择测试集*******
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    ## *******随机选取训练集*******
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('the Classification error is:', docList[docIndex])
    rate = float(errorCount) / len(testSet)
    print('the error rate is :', rate)
    return rate

# RSS源分类器及高频词去除函数
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = lambda item:item[1], \
                        reverse= True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList = []
    classList = []
    fullText =[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    print(docList, fullText, classList)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key = lambda pair:pair[1], reverse= True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key= lambda pair:pair[1], reverse= True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY:
        print(item[0])




# mean_rate = []
# for i in range(10):
#     rate = spamTest()
#     mean_rate.append(rate)
# print(sum(mean_rate)/10)

# ny = feedparser.parse('http://feeds.sciencedaily.com/sciencedaily')
# sf = feedparser.parse('http://feeds.nature.com/nature/rss/current')
# vocabList, psF, pNY = localWords(ny, sf)
# getTopWords(ny, sf)