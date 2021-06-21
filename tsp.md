---
title: tsp
categories:
  - java学习
  - python学习
  - 开发项目
  - 论文
  - 电脑维修
  - 金融
  - 乐器
  - 总结
  - 每日
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-06-21 20:45:48
urlname:
tags:
---

<center><font size=7>遗传算法解决TSP问题</font></center>

<center><font size=4>叶帅3120305696</font></center>

 1.题目描述

​	TSP问题（旅行商问题）是指旅行家要旅行n个城市，要求各个城市经历且仅经历一次然后回到出发城市，并要求所走的路程最短。

 

   转换为图论问题即：给定一个 n 个节点的加权完全图 d，求一权重最小的哈密尔顿回路p。

2.实验设计

​    	条件符号：

​			1.总共N个节点，节点用数字编号 0,1,…,n-1 表示

​			2.任意两个节点距离用邻接矩阵 d表示,其中i，j∈[0,N]

​		遗传算法符号

​			种群规模 M,

​			路径p[i][0:N-1] 表示种群中第i个元素所对应的路径[0:N-1]

​			路径长度l[i] 表示种群中第i个元素所对应的路径长度

​			变异概率Pm = 0.1= 10%

​		特殊符号

​			为了简单，不设置坐标，只使用邻接图来表示城市地图。

3. 算法思想

   所有ga算法都可以是这种设计

   ![GA](D:\git_rep\hexo\source\_posts\tsp\2019021212002269 (1).png)

杂交：

杂交分两步，第一步是适应度的计算，程序里用cal_adp()函数实现，第二步是杂交，让适应度越高的个体杂交的几率越大。

1. **适应度函数计算：cal_adp()**

   适应度函数主要是给**杂交**操作提供数据，以便适应度越高的**杂交**计算当前种群的。主要的步骤就是：计算路径长度$l[i]$，取适应度$p[i] = frac{1}{l[i]}$,然后归一化

   $p[i] = frac{（p[i]）}{Sigma  （p[i]）}$ ,将适应度作为杂交被选取到的概率。

2. 杂交

   根据适应度选取两个不同个体，然后各自作为父亲，取前一半路径，剩下一半取另外一个个体的剩余路径，生成两个子代。

   进行$frac{M}{2}$次这样的操作，产生$M$个子代。

变异：mutate

变异就是将杂交后生成的子代中每一个进行一个变异操作：若随机数小于变异概率，随机调换该个体中的两个节点。

选择：

生成下一代，这里不将杂交变异后的子代直接作为下一代，





4.代码

```
import numpy as np
import random
import copy
import math
import sys
import pylab as pl

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

class GA():
    def __init__(self, N, M, randseed=None):  
        self.N = N
        self.M = M
        self.seed = randseed
        # d 距离邻接矩阵
        self.d = []
        self.path = []
        self.newpath = []
        self.l = []
        self.p = []
        self.bestl = sys.maxsize
        # 杂交概率 pc 变异概率 pm
        self.pm = 0.9
        self.pc = 0.2

        # 生成 d 距离邻接矩阵
        np.random.seed(self.seed)
        self.d = np.random.randint(1, 1000, (self.N, self.N))
        self.d = np.triu(self.d, 1) + np.triu(self.d, 1).T  # 生成完全图

        # 生成一个含M个个体的初始种群
        x = [i for i in range(self.N)]
        for i in range(self.M):
            self.path.append(copy.copy(x))
            random.shuffle(x)
        # print('init')
        # print('d 距离邻接矩阵')
        #print(self.d)
        # print('初始路径距离矩阵')
        #print(self.path)

    def cal_adp(self):  # 计算适应度 储存当前代最优值
        # 路径 path[i][:]
        # 路径长度 l[i]
        # 归一化适应度 adp[i]
        # 生成一个含M个个体的初始种群
        # 生成路径 path[i][:] 第i个个体路径
        # 生成路径长度 l[i] 第i个个体路径长度
        self.l = []
        self.p = []
        self.bestl = sys.maxsize
        self.bestpath = []
        for i in range(self.M):
            length = 0
            for j in range(self.N - 1):
                length += self.d[self.path[i][j]][self.path[i][j + 1]]
            length += self.d[self.path[i][self.N - 1]][self.path[i][0]]
            self.l.append(copy.copy(length))

        # 生成归一化适应度 p[i]  储存当前代最优值
        for i in range(self.M):
            self.p.append(math.pow(self.l[i], -1))
            if self.bestl > self.l[i]:
                self.bestl = copy.copy(self.l[i])
                self.bestpath = copy.deepcopy(self.path[i])
                # print('l[i]',self.l[i])
        sump = math.fsum(self.p)
        for i in range(self.M):
            self.p[i] /= sump

    def hybrid(self):  # 杂交
        self.path = sorted(self.path, key=lambda i: self.p[self.path.index(i)])
        self.p = sorted(self.p)
        # print('杂交前path')
        # print(np.array(self.path))
        for i in range(1, self.M):
            self.p[i] += self.p[i - 1]
        # print(p)
        self.newpath = []
        for i in range(self.M // 2):  # 进行M/2次杂交 一共生成M个子代
            x, y = 0, 0
            while x == y:
                a, b = np.random.rand(), np.random.rand()
                x, y = 0, 0
                while a > self.p[x]:
                    x += 1
                while b > self.p[y]:
                    y += 1
            # x作为父代
            child = self.path[x][:self.N // 2]
            child2 = list(set(self.path[y]) - set(child))
            child2.sort(key=self.path[y].index)
            # print(self.path[x],self.path[y])
            # print(child,child2)
            self.newpath.append(child + child2)

            # y作为父代
            child = self.path[y][:self.N // 2]
            child2 = list(set(self.path[x]) - set(child))
            child2.sort(key=self.path[x].index)
            # print(self.path[x],self.path[y])
            # print(child,child2)
            self.newpath.append(child + child2)

        # print('杂交后newpath')
        # print(np.array(self.newpath))

    def mutate(self):  # 变异
        # print(np.array(self.path))
        for i in range(self.M):
            if np.random.rand() < self.pm:  # 概率变异
                a, b = 0, 0
                while a == b:
                    a, b = np.random.randint(0, self.N), np.random.randint(0, self.N)
                self.newpath[i][a], self.newpath[i][b] = self.newpath[i][b], self.newpath[i][a]
                # print(i,a,b)

    def generate(self):  # 生成下一代 将M个子代 和 M个父代合并筛选 出M个子代的新种群
        # print(np.array(self.newpath))
        # print(np.array(self.path))
        # t = self.newpath +self.path[:self.M//2]
        t = self.newpath
        l = []
        for i in range(int(self.M)):
            length = 0
            for j in range(self.N - 1):
                length += self.d[t[i][j]][t[i][j + 1]]
            length += self.d[t[i][self.N - 1]][t[i][0]]
            l.append(copy.copy(length))
        t = sorted(t, key=lambda i: l[t.index(i)])
        l = sorted(l)
        #print('l[0]',l[0])
        # if l[0] < self.l[0]:
        #     t[self.M-1] = copy.copy(self.path[self.M-1])
        t[np.random.randint(0,self.M)] = copy.copy(self.path[self.M-1])
        self.path = t[:self.M]  # 切取最优秀的前M个成为子代path
        # print('新子代path')
        # print(np.array(self.path))


if __name__ == '__main__':
    tsp = GA(N=30, M=1000, randseed=1)  # 初始化
    ans = sys.maxsize
    log = []
    stp = 0
    tim = 0
    length_save=[]
    while stp <= 1000:  # 100代最优解没变化则停止
        tim += 1  # 记录生成次数
        tsp.cal_adp()  # 计算适应度
        log.append(tsp.bestl)
        # print(ans)
        if tsp.bestl == ans:
            stp += 1
        elif tsp.bestl != ans:
            ans = tsp.bestl
            stp = 0
        tsp.hybrid()  # 杂交
        tsp.mutate()  # 变异
        tsp.generate()  # 生成子代
        length_save.append(tsp.bestl)

        print(tsp.bestl,tim)
    print('Path:', tsp.bestpath)
    print('Length:', tsp.bestl)
    print('Times', tim)
    times = np.arange(0,tim)
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

    plt.xlabel("生成次数", fontproperties=font_set)
    plt.ylabel("最短路径", fontproperties=font_set)

    plt.plot(times, length_save)

    plt.show()

```



4.实验结果



pc=pm=0.2

![Figure_1](C:\Users\ye97\Desktop\Figure_1.png)



