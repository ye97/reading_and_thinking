---
title: 传统paper
categories:
  - 论文
tags:
  - 传统

date: 2021-03-13 10:16:46
---



# 1，逐步求精点云配准

## 1,目标函数

$$
\begin{array}{c}
\underset{\left\{R_{i}, \boldsymbol{t}_{i}\right\}_{i=2}^{N}, \boldsymbol{q}_{c(i j)} \in Q_{i}}{\arg \min } \sum_{i=2}^{N} \sum_{j=1}^{M_{i}}\left(w_{i j} \| R_{i} \boldsymbol{p}_{i j}+\right. 
\left.\boldsymbol{t}_{i}-\boldsymbol{q}_{c(i j)} \|_{2}^{2}\right)

\end{array}
$$


注释：P表示初始点云model
$$
\begin{array}
 P =\left\{R_{i}^{0} \boldsymbol{p}_{i j}+\boldsymbol{t}_{i}^{0}\right\}_{i=1, j=1}^{N, M_{j}}

\end{array}
$$
以及不完整模型Q:A\B表示在集合A而不在集合B中的元素，就是说删除第i帧点云
$$
Q_{i}=P \backslash\left\{R_{i}^{0} \boldsymbol{p}_{i j}+\boldsymbol{t}_{i}^{0}\right\}_{j=1}^{M_{i}}
$$


qc表示点i，j形成的点对
$$
\begin{equation}
qc(i,j)

\end{equation}
$$
w表示权重：
$$
\begin{equation}
w_{i j}

\end{equation}
$$

## 2,算法步骤：

1) 根据初始值 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right)$, 构造初始模型 P.  

2) 顺序遍历基准帧以外的每帧点云, 针对第 i帧点云: a) 利用有效的双视角配准方法计算 Pi 与Qi 之间的最新刚体变换关系 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right)$.b) 根据当前计算获得的最新参数 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right) $更新模型 P.  

3) 重复执行步骤 2), 直到满足循环停止条件.当循环次数 k 超过设定阈值 K 或前后两次循环所求的刚体变换变化小于设定的阈值时, 即可停止循环, 输出多视角点云配准结果.  

## 3,权重迭代最近点算法

​	在逐步求精的第 k 次循环过程中, Pi 与 Qi 之间的最新刚体变换关系 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right) $ 需要通过求解以下最小二乘问题后得到  

1) 建立点云 Pi 与模型 Qi 之间的点对关系  $Q_{i}=P \backslash\left\{R_{i}^{0} \boldsymbol{p}_{i j}+\boldsymbol{t}_{i}^{0}\right\}_{j=1}^{M_{i}}$

2) 为所建立的最新点对计算相应的权重  $w_{i j, k}=\alpha_{i j, k} \exp \left(-\frac{d_{i j, k}^{2}}{2 \sigma^{2}}\right)$

其中w权重为：$\alpha_{i j, k}=\left\{\begin{array}{ll}1, & \boldsymbol{q}_{i j, k} \in P_{1} \\ 0.5, & \boldsymbol{q}_{i j, k} \notin P_{1}\end{array}\right.$

 3) 根据最新点对及其权重, 计算最优刚体变换  $\left(w_{i j} \| R_{i} \boldsymbol{p}_{i j}+\right. 
\left.\boldsymbol{t}_{i}-\boldsymbol{q}_{c(i j)} \|_{2}^{2}\right)$



## 4,单帧更新算法

![逐帧算法](D:\git_rep\hexo\source\_posts\传统paper\逐帧算法.png)

## 5，算法评估

​     旋转矩阵误差：$e_{R}=\frac{1}{N} \sum_{i=1}^{N}\left\|R_{i, m}-R_{i, g}\right\|_{F}$

​	平移向量误差 ：$e_{t}=\frac{1}{N} \sum_{i=1}^{N}\left\|\boldsymbol{t}_{i, m}-\boldsymbol{t}_{i, g}\right\|_{2}$

## 6,参考算法

​	Low-rank and sparse matrix decomposition, LRS  

​	MA



# 2，MATRICP

## 1,目标函数

$\min _{\xi, \mathbf{R}, \vec{t}}\left(\frac{1}{\left|P_{\xi}\right| \xi^{1+\lambda}} \sum_{\vec{p}_{a} \in P_{\xi}}\left\|\mathbf{R} \vec{p}_{a}+\vec{t}-\vec{q}_{c(a)}\right\|_{2}^{2}\right)$s.t. $\quad \mathbf{R}^{T} \mathbf{R}=\mathbf{I}_{3}, \quad \operatorname{det}(\mathbf{R})=1$$\quad \xi \in\left[\xi_{\min }, 1\right], P_{\xi} \subseteq P, \quad\left|P_{\xi}\right|=\xi|P|$

$P_{\xi}$表示配准子集

## 2，算法步骤

$$ c_{k}(a)=\underset{b \in\left\{1,2, . ., N_{q}\right\}}{\arg \min }\left\|\mathbf{R}_{k-1} \vec{p}_{a}+\vec{t}_{k-1}-\vec{q}_{b}\right\|_{2} $$

# the end：代码学习总结

## 1，数据读取问题

### 	1.1先通过matlab查看数据结构

![维度](D:\git_rep\hexo\source\_posts\传统paper\维度.png)

### 1.2在python中加载数据

~~~python
from scipy import io as sio
 dataset = sio.loadmat(path)
#dataset是字典形式
#上图是他的键
#读取键，debug查看具体的结构
model = dataset.get('model')


~~~

### 1.3寻找最近点对（注意scilearn 官方文档使用）

~~~python
#sklearn包调用
from sklearn.neighbors import NearestNeighbors
#构造最近搜索类
searcher = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
#训练类（train medel）
searcher.fit(modelI[:3, :].T)
#返回数组
rs = searcher.kneighbors(TData[:3, :].T)
#
~~~

~~~python
samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(n_neighbors=1)
>>> neigh.fit(samples)
NearestNeighbors(n_neighbors=1)
>>> print(neigh.kneighbors([[1., 1., 1.]]))
#第一个数据表示对应点的distance 第二数组表示index
(array([[0.5]]), array([[2]]))
~~~



### the end 常见命名总结

grt=ground truth 

iterNum 

scan 帧（一般是处理后的点云（兔子模型10帧）），scan下面添加1111，方便和p相乘

shape 原点云，就是模型的形状

p 变换矩阵，4 乘4，下面是0001，上面是RT



