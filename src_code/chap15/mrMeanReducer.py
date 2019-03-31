#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019/1/5 22:05
# @Author  : Despicable Me
# @Email   : 
# @File    : mrMeanReducer.py
# @Software: PyCharm
# @Explain :
"""
*************************
分布式均值和方差计算的reducer：
mapper 通过对接收的原始输入数据进行处理，产生中间值(这里为数据元素的个数，均值，元素平方的均
值),并将中间值传递给reducer,由于mapper式并行执行的，故需要将这些mapper的输出合并为一个值。
即：将中间的key/value对进行组合
*************************
"""
print(__doc__)

import sys
from numpy import mat, mean, power

def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)                     # 创建一个输入的数据行的列表list

mapperout = [line.split('\t') for line in input]
cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0
for instance in mapperout:                        # 遍历样本中所有的数据
    nj = float(instance[0])                       # nj为每组数据包含的样本个数
    # 可以理解为由于有多个mapper,每个mapper返回一组数据(单个样本的元素个数，均值，元素平方的均值)
    # 因此reducer接收多组数据，并且reducer根据计算出的多组数据，计算出总样本的个数，元素的总和，样本元素的平方和
    cumN += nj                                    # 累计样本个数的总和
    cumVal += nj * float(instance[1])             # 累计样本元素的总和
    cumSumSq += nj * float(instance[2])           # 累计样本中元素的平方和
mean = cumVal / cumN                              # 总样本的均值
varSum = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean)/cumN  # 总样本的方差，这里为样本方差表达式的分解式
print('%d\t%f\t%f' % (cumN, mean, varSum))        # 样本总个数， 均值，方差，即全局均值，方差
print('report: still alive', file= sys.stderr)