#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @Time    : 2018/12/24 16:42
# @Author  : Despicable Me
# @Email   : 
# @File    : treeExplore.py
# @Software: PyCharm
# @Explain :

from numpy import *
import tkinter as tk
from regTrees import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    '''
    函数说明： 绘制树
    :param tolS:
    :param tolN:
    :return:
    '''
    reDraw.f.clf()                         # 清空之前的图像，各个子图也会被清空
    reDraw.a = reDraw.f.add_subplot(111)   # 重新绘制子图
    if chkBtnVar.get():                    # 检查复选框是否被选中
        if tolN < 2:                       # 成立时，构建模型树
            tolN = 2
        myTree = createTree(reDraw.rawDat, modelLeaf,\
                                modelErr, (tolS, tolN))
        yHat = createForceCast(myTree, reDraw.testDat, modelTreeEval)
    else:                                 # 不成立，构建回归树
        myTree = createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = createForceCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0].A,reDraw.rawDat[:,1].A, s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    try:                               # 如果能把输入文本解析成整数，继续执行
        tolN = int(tolNentry.get())    # 首先从tolNentry(输入框)中获取输入的数据，如果没有输入
    except:                            # 如果不能把输入文本解析成整数，执行except,即默认设定值
        tolN = 10
        print('entry Integer for tolN')
        tolNentry.delete(0, END)       # 清空输入框
        tolNentry.insert(0,'10')       # 回复默认值
    try:
        tolS = float(tolSentry.get())  # 首先从tolSentry(输入框)中获取输入的数据，如果没有输入
    except:                            # 如果不能把输入文本解析成浮点数，执行except,即默认设定值
        tolS = 1.0
        print('entry float for tolS')
        tolSentry.delete(0, END)       # 清空输入框
        tolSentry.insert(0,'1.0')      # 恢复默认值
    return tolN, tolS                  # 返回tolN,tolS

def drawNewTree():
    tolN, tolS = getInputs()           # 获得(输入框)中的tolN,tolS
    reDraw(tolS, tolN)

# # main 函数
root = tk.Tk()

tk.Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)

# tolN
tk.Label(root, text='tolN').grid(row=1, column=0)  # 创建tolN标签
tolNentry = tk.Entry(root)                         # 创建对应的输入框
tolNentry.grid(row=1,column=1)                     # grid 网格布局管理器
tolNentry.insert(0,'10')
# tolS
tk.Label(root, text='tolS').grid(row=2, column=0)  # 创建tolS标签
tolSentry = tk.Entry(root)                         # 创建对应的输入框
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')                          #
# Button
tk.Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)  # 创建按钮
# 按钮整数值
chkBtnVar = tk.IntVar()          # InVar 按钮整数值，chkBtnVar为为读取checkbutton的状态
# 按钮复选框
chkBtn = tk.Checkbutton(root, text='Model Tree', variable = chkBtnVar)   # checkbutton创建复选框按钮
chkBtn.grid(row=3, column=0, columnspan=2)

# 创建画板 canvas
reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master= root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
# 初始化数据
reDraw.rawDat = mat(loadDataSet('D:\Projects_Python\Dong\GitHub Files\AiLearning\db\9.RegTrees\sine.txt'))              # 读取数据
reDraw.testDat = arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:,0]), 0.01)
reDraw(1.0, 10)

root.mainloop()

