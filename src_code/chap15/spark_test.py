# !/usr/bin/env python
# -*- coding:utf-8 -*-

# @Time    : 2019/1/6 16:29
# @Author  : Despicable Me
# @Email   : 
# @File    : spark_test.py
# @Software: PyCharm
# @Explain : mrjob中分布式Pegasos算法的外围代码

# from pyspark import SparkContext
#
# from pyspark import SparkContext as sc
# from pyspark import SparkConf
#
# conf = SparkConf().setAppName('miniProject').setMaster('local[*]')
# sc = SparkContext.getOrCreate(conf)
# rdd = sc.parallelize([1,2,3,4,5])
# rdd
# print(rdd)
# print(rdd.getNumPartitions())
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster('local').setAppName('MY First App')
sc = SparkContext(conf = conf)
# sc.setLogLevel()