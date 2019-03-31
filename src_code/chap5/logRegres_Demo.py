#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 22:44
# @Author  : Despicable Me
# @Site    : 
# @File    : logRegres_Demo.py
# @Software: PyCharm
# @Explain :
from logRegres import *
from numpy import *
dataArr, labelmat = loadDataSet()
weights = gradAscent(dataArr, labelmat)
print(weights)
plotBestFit(weights)

# weights , dataOfCoe= stocGradAscent1(array(dataArr), labelmat)
# plotBestFit(weights)

# plotTheCoeff(array(dataOfCoe))

multiTest()