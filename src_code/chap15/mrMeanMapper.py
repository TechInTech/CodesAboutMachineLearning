#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2019/1/4 23:08
# @Author  : Despicable Me
# @Email   : 
# @File    : mrMeanMapper.py
# @Software: PyCharm
# @Explain :
import sys
from numpy import mat, mean, power

"""
分布式均值和方差计算的mapper:
对读入的数据求取数据个数，得到所有数据的平均值，以及数据
中所有元素的平方的均值，这些值将用来计算全局的均值和方差
"""

def read_input(file):
    for line in file:
        yield line.rstrip()    # rstrip()用删除字符串末尾的空格,返回一个yield迭代器，每次获得下一个值，节约内存
    # yield在函数中的功能类似于return，不同的是yield每次返回结果之后函数并没有退出,
    # 而是每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用
    # 如果一个函数需要多次循环执行一个动作，并且每次执行的结果都是需要的，这种场景很适合使用yield实现
    # 包含yield的函数成为一个生成器，生成器同时也是一个迭代器，支持通过next方法获取下一个值


input = read_input(sys.stdin)             # 创建一个输入的数据行的列表list
input = [float(line) for line in input]   # 将数据全部转化为float类型
numInputs = len(input)                    # 得到数据的个数，即输入文本的数据的行数
input = mat(input)                        # 将列表转化为矩阵
sqInput = power(input, 2)                 # 求得矩阵各个元素的平方

print('%d\t%f\t%f' % (numInputs, mean(input), mean(sqInput)))   # 输出数据的个数，所有数据的平均值，所有数据平方和的平均值
print('report: still alive', file= sys.stderr)                  # 标准错误输出，即对主节点作出的响应报告，表明本节点工作正常。
                                                                # 注意：一个好的习惯是想标准错误输出发送报告。如果某任务10分钟内没有报告输出，则将被Hadoop中止。