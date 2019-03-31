#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/27 19:12
# @Author  : Despicable Me
# @Site    : 
# @File    : bayes_Demo.py
# @Software: PyCharm
# @Explain :

from bayes import *
# spamTest()
listOPosts, listClasses = loadDataSet()
myVocablist = createVocabList(listOPosts)
print(myVocablist)
set2V = setOfWords2Vec(myVocablist, listOPosts[0])
print(set2V)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
print(pAb, '\n', p0V, '\n', p1V)
vec2Classify = setOfWords2Vec(myVocablist, ['2', 'S'])
print(vec2Classify)
a = classifyNB(vec2Classify, p0V, p1V, pAb)
print(a)