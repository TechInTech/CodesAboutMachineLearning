#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
************************************
# @Time    : 2019/1/6 10:21
# @Author  : Despicable Me
# @Email   : 
# @File    : mrMean.py
# @Software: PyCharm
# @Explain : 分布式均值方差计算的mrjob实现
************************************
"""
print(__doc__)

from mrjob.job import MRJob
from mrjob.step import MRStep


class MRmean(MRJob):
    """
    分布式地计算均值和方差。输入文本分发给很多mappers俩计算中间值，
    这些中间值再通过reducer进行累加，从而计算出全局的均值和方差。
    """
    def __init__(self, *args, **kwargs):
        super(MRmean, self).__init__(*args, **kwargs)
        self.inCount = 0
        self.inSum = 0
        self.inSqSum = 0


    def map(self, key, val):                 # 接收输入数据流；key,val键值对
        """
        在每行输入上执行同样的步骤，对输入进行累积
        """
        if False: yield
        inVal = float(val)                   # 将数据转化为浮点数
        self.inCount += 1                    # 总数加1
        self.inSum += inVal                  # 求和
        self.inSqSum += inVal * inVal        # 平方和


    def map_final(self):
        """
        待map中所有值收集完毕后计算出均值和平方均值，最后将这些值作为中间值通过yield语句传递出去
        """
        mn = self.inSum/self.inCount         # 样本均值
        mnSq = self.inSqSum/self.inCount     # 样本平方和的均值
        yield (1, [self.inCount, mn, mnSq])  # 返回样本数，样本均值，样本平方和的均值(迭代器)


    def reduce(self, key, packdValues):      # 返回全局均值和方差，reduce的输入存放在迭代器对象中
        cumVal, cumSumSq, cumN = 0.0, 0.0, 0.0
        for valArr in packdValues:
            nj = float(valArr[0])
            cumN += nj
            cumVal += nj * float(valArr[1])
            cumSumSq += nj * float(valArr[2])
        mean = cumVal/cumN                   # 样本均值
        var = (cumSumSq - 2 * mean * cumVal + cumN * mean * mean)/cumN # 样本方差
        yield (mean, var)


    def steps(self):
        """
        定义了执行的步骤
        """
        return ([MRStep(mapper= self.map, reducer= self.reduce, mapper_final= self.map_final)])

if __name__ == '__main__':
    MRmean.run()