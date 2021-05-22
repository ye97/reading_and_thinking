---
title: 传统paper
categories:
  - 论文
tags:
  - 传统

date: 2021-03-13 10:16:46
description: <read more ...>
mathjax: true
typora-root-url: 传统paper
---

# 逐步求精点云配准

## 目标函数

​																	$$\begin{array}{c}
\underset{\left\{R_{i}, \boldsymbol{t}_{i}\right\}_{i=2}^{N}, \boldsymbol{q}_{c(i j)} \in Q_{i}}{\arg \min } \sum_{i=2}^{N} \sum_{j=1}^{M_{i}}\left(w_{i j} \| R_{i} \boldsymbol{p}_{i j}+\right. 
\left.\boldsymbol{t}_{i}-\boldsymbol{q}_{c(i j)} \|_{2}^{2}\right) \end{array}$$


注释：P表示初始点云model

​																<center>$p\begin{array}P =\left\{R_{i}^{0} \boldsymbol{p}_{i j}+\boldsymbol{t}_{i}^{0}\right\}_{i=1, j=1}^{N, M_{j}}\end{array}$</center>

以及不完整模型Q:A\\B表示在集合A而不在集合B中的元素，就是说删除第i帧点云

​															<center>	$Q_{i}=P \backslash\left\{R_{i}^{0} \boldsymbol{p}_{i j}+\boldsymbol{t}_{i}^{0}\right\}_{j=1}^{M_{i}}$  </center>


qc表示点i，j形成的点对	:	$\begin{equation}qc(i,j)\end{equation}$





w表示权重：

​																$\begin{equation}w_{i j}\end{equation}$

## 算法步骤：

1) 根据初始值 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right)$, 构造初始模型 P.  

2) 顺序遍历基准帧以外的每帧点云, 针对第 i帧点云: a) 利用有效的双视角配准方法计算 Pi 与Qi 之间的最新刚体变换关系 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right)$.b) 根据当前计算获得的最新参数 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right) $更新模型 P.  

3) 重复执行步骤 2), 直到满足循环停止条件.当循环次数 k 超过设定阈值 K 或前后两次循环所求的刚体变换变化小于设定的阈值时, 即可停止循环, 输出多视角点云配准结果.  

## 权重迭代最近点算法

​	在逐步求精的第 k 次循环过程中, Pi 与 Qi 之间的最新刚体变换关系 $\left(R_{i}^{0}, \boldsymbol{t}_{i}^{0}\right) $ 需要通过求解以下最小二乘问题后得到  

1) 建立点云 Pi 与模型 Qi 之间的点对关系  $Q_{i}=P \backslash\left\{R_{i}^{0} \boldsymbol{p}_{i j}+\boldsymbol{t}_{i}^{0}\right\}_{j=1}^{M_{i}}$

2) 为所建立的最新点对计算相应的权重  $w_{i j, k}=\alpha_{i j, k} \exp \left(-\frac{d_{i j, k}^{2}}{2 \sigma^{2}}\right)$

其中w权重为：$\alpha_{i j, k}=\left\{\begin{array}{ll}1, & \boldsymbol{q}_{i j, k} \in P_{1} \\ 0.5, & \boldsymbol{q}_{i j, k} \notin P_{1}\end{array}\right.$

 3) 根据最新点对及其权重, 计算最优刚体变换  $\left(w_{i j} \| R_{i} \boldsymbol{p}_{i j}+\right. 
\left.\boldsymbol{t}_{i}-\boldsymbol{q}_{c(i j)} \|_{2}^{2}\right)$



## 单帧更新算法

![](逐帧算法.png)

## 算法评估

​     旋转矩阵误差：$e_{R}=\frac{1}{N} \sum_{i=1}^{N}\left\|R_{i, m}-R_{i, g}\right\|_{F}$

​	平移向量误差 ：$e_{t}=\frac{1}{N} \sum_{i=1}^{N}\left\|\boldsymbol{t}_{i, m}-\boldsymbol{t}_{i, g}\right\|_{2}$

## 参考算法

​	Low-rank and sparse matrix decomposition, LRS  

​	MA

## icp算法求解

​			![定义质心](定义质心.svg)

​			![定义去质心](定义去质心.svg)

![](icp优化函数.svg)

​											![简化icp](简化icp.svg)

![icp证明](icp证明.svg)由于质心的旋转性质，这部分为0，关于t的部分直接令他为0，R的求解需要记一下														![最终优化函数](最终优化函数.svg)

​			1.根据两组已知的点集算出 $ W=\sum_{i=1}^{n} q_{i} q_{i}^{\prime T} $,$ 并对其做奇异值分解得到 $  $W=U \Sigma V^{T}$ 。

2. 计算 $R^{*}=U V^{T}$ 。
3. 计算 $t^{*}=p-R^{*} p^{\prime}$





# ICP算法证明

## 问题引入

        代最近点（Iterative Closest Point, 下简称ICP）算法是一种点云匹配算法。

        假设我们通过RGB-D相机得到了第一组点云 ![P = \left \{p _{1} ,p _{2} ,p _{3} ,\cdots ,p _{n} \right \}](/1620608087-7b6d2e32bd13e8873a5916ffd41a647f.latex)，相机经过位姿变换（旋转加平移）后又拍摄了第二组点云![Q = \left \{q _{1} ,q _{2} ,q _{3} ,\cdots ,q _{n} \right \}](/1620608087-686130d166c67964f8d2281dce493688.latex) ,注意这里的 ![P](/1620608087-6325ac7a6a3ea0b7773f7bac03a8f653.latex) 和 ![Q](/1620608087-91d4657f758d80796061a4d7d476ba60.latex) 的坐标分别对应移动前和移动后的坐标系（即坐标原点始终为相机光心，这里我们有移动前、移动后两个坐标系），并且我们通过相关算法筛选和调整了点云存储的顺序，使得 ![P](/1620608087-6325ac7a6a3ea0b7773f7bac03a8f653.latex)和![Q](/1620608087-91d4657f758d80796061a4d7d476ba60.latex) 中的点一一对应，如$$ [\left (P _{99},Q _{99} \right )]$$在三维空间中对应同一个点。

        现在我们要解决的问题是：计算相机的旋转  ![R](/1620608087-ddfe2a175e23e49d70fd49f77c4c8221.latex) 和平移 ![t](/1620608087-b5eeaedf6ad956f27dfd8421ea1c8e45.latex) ，在没有误差的情况下，从![P](/1620608087-6325ac7a6a3ea0b7773f7bac03a8f653.latex) 坐标系转换到![Q](/1620608087-91d4657f758d80796061a4d7d476ba60.latex) 的公式为：

                                              ![q_{i} = Rp_{i} + t](/1620608087-34738130a734b48eccef5c3f7c970e10.latex)

        但由于噪声及错误匹配（如$$ [\left (P _{99},Q _{99} \right )]$$)其实并不对应空间中同一点，但特征匹配算法错误地认为二者是同一点）的存在， 上式不总是成立，所以我们要最小化的目标函数为

                                             ![\frac{1}{2}\sum_{i = 1}^{n}\left \| q_{i} - Rp_{i} - t \right \|^{2}](/../%25E4%25BC%25A0%25E7%25BB%259Fpaper/1620608087-20f16dbe751a2c3239f7ec5202719ef1.latex)

        常用的求解 ![R](/../%25E4%25BC%25A0%25E7%25BB%259Fpaper/1620608087-ddfe2a175e23e49d70fd49f77c4c8221.latex) 和 ![t](/../%25E4%25BC%25A0%25E7%25BB%259Fpaper/1620608087-b5eeaedf6ad956f27dfd8421ea1c8e45.latex) 的方法有：

1.  SVD
2.  非线性优化

        非线性优化的描述比较繁琐，下面只介绍SVD方法。为了后面能使用SVD，我们需要在数学上做一点小小的变形。首先定义两组点云的质心（center of mass）为 ![\mu _{p} = \frac{1}{n} \sum_{i = 1}^{n}p_{i}](/../%25E4%25BC%25A0%25E7%25BB%259Fpaper/1620608087-9f248482c67c244b2fc9e39472d608c8.latex),![\mu _{q} = \frac{1}{n} \sum_{i = 1}^{n}q_{i}](/../%25E4%25BC%25A0%25E7%25BB%259Fpaper/1620608087-eae2041b5e83420896ec3aa6b677ac51.latex),并作出如下处理：

![](/../%25E4%25BC%25A0%25E7%25BB%259Fpaper/1620608087-36ae0bb46b8b434797ba6bce09d9664e.png)

![](/../%25E4%25BC%25A0%25E7%25BB%259Fpaper/1620608087-ab16695bc54b37f5da0d891ab02673ed.png)

此处r等于UV参见论文，简单得说就是保证秩最大。

此处要记住得就是R的求解定式。



# MATRICP

## 目标函数

$\min _{\xi, \mathbf{R}, \vec{t}}\left(\frac{1}{\left|P_{\xi}\right| \xi^{1+\lambda}} \sum_{\vec{p}_{a} \in P_{\xi}}\left\|\mathbf{R} \vec{p}_{a}+\vec{t}-\vec{q}_{c(a)}\right\|_{2}^{2}\right)$  s.t. $\quad \mathbf{R}^{T} \mathbf{R}=\mathbf{I}_{3}, \quad \operatorname{det}(\mathbf{R})=1$$\quad \xi \in\left[\xi_{\min }, 1\right], P_{\xi} \subseteq P, \quad\left|P_{\xi}\right|=\xi|P|$

$P_{\xi}$表示配准子集

## 算法步骤

李群关系：

$ c_{k}(a)= \underset{b \in\left\{1,2, . ., N_{q}\right\}}{\arg \min }\left\|\mathbf{R}_{k-1} \vec{p}_{a}+\vec{t}_{k-1}-\vec{q}_{b}\right\|_{2} $



![](/v2-11bc566d92c36f9ac7bae27dee7edf8e_1440w.jpg)

目标函数tricp：

![](/%E5%9B%BE%E7%89%871.png)

![图片2](/%E5%9B%BE%E7%89%872.png)

![](/%E5%9B%BE%E7%89%873.png)

![](/%E5%9B%BE%E7%89%874.png)

![](/%E5%9B%BE%E7%89%875.png)

权重matricp

![](/%E5%9B%BE%E7%89%876.png)





# LRS

# AA-ICP

## 	AA基本变种

![Aderson accelerate variant](Adersonacceleratevariant.png)

## 		AA-ICP

​											<img src="/AA-ICP.png" alt="AA-ICP" style="zoom:100%;" />		

​		

# RANSAC

**Random sample consensus** (**RANSAC**) 

---



```python
Given:
    data – A set of observations.
    model – A model to explain observed data points.
    n – Minimum number of data points required to estimate model parameters.
    k – Maximum number of iterations allowed in the algorithm.
    t – Threshold value to determine data points that are fit well by model.
    d – Number of close data points required to assert that a model fits well to data.

Return:
    bestFit – model parameters which best fit the data (or null if no good model is found)

iterations = 0
bestFit = null
bestErr = something really large

while iterations < k do
    maybeInliers := n randomly selected values from data
    maybeModel := model parameters fitted to maybeInliers
    alsoInliers := empty set
    for every point in data not in maybeInliers do
        if point fits maybeModel with an error smaller than t
             add point to alsoInliers
    end for
    if the number of elements in alsoInliers is > d then
        // This implies that we may have found a good model
        // now test how good it is.
        betterModel := model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr := a measure of how well betterModel fits these points
        if thisErr < bestErr then
            bestFit := betterModel
            bestErr := thisErr
        end if
    end if
    increment iterations
end while

return bestFit
```



# 李佳元-Point Cloud Registration Based on One-Point RANSAC and Scale-Annealing Biweight Estimation

通过分解问题为r，t，s参数分别求解。

## RANSAC算法次数：

​																							$$ 		N_{T}=\left[\frac{\log (1-p)}{\log \left(1-(1-p)^{m}\right)}\right] $$																			（3）

其中p是良好子集的置信度，通常设置为0.99； m是最小子集的大小（对于7参数配准，m = 3）；

## Line Vector:

给定点对$\left(\boldsymbol{x}_{i}, \boldsymbol{y}_{i}\right)$ and $\left(\boldsymbol{x}_{j}, \boldsymbol{y}_{j}\right)$, 线向量$\left(\vec{x}_{i j}=x_{i}-x_{j}, \vec{y}_{i j}=y_{i}-y_{j}\right) .$ 如果 $\left(x_{i}, y_{i}\right)$ and $\left(x_{j}, y_{j}\right)$
都是内值
$$
\begin{aligned}
\vec{y}_{i j} &=s \mathbf{R}\left(x_{i}-x_{j}\right)+\left(n_{i}-n_{j}\right) \\
&=s \mathbf{R} \vec{x}_{i j}+\vec{n}_{i j}                                          （4）													
\end{aligned}
$$
$\overrightarrow{\boldsymbol{n}}_{i j}=\boldsymbol{n}_{i}-\boldsymbol{n}_{j}$ 噪声向量. 噪声边界 $\tau$, $\left\|\overrightarrow{\boldsymbol{n}}_{i j}\right\| \leq 2 \tau$. （4）式消除了t变量，只和r和s有关.

对 (4) 进行二范数：
$$
\left\|\vec{y}_{i j}\right\|=\left\|s \mathbf{R} \vec{x}_{i j}+\vec{n}_{i j}\right\| .
$$
三角绝对值不等式得到：
$$
-\left\|\overrightarrow{\boldsymbol{n}}_{i j}\right\| \leq\left\|\overrightarrow{\boldsymbol{y}}_{i j}\right\|-\left\|s \mathbf{R} \overrightarrow{\boldsymbol{x}}_{i j}\right\| \leq\left\|\overrightarrow{\boldsymbol{n}}_{i j}\right\| （6)
$$
(6) i左右除以$\left\|\vec{x}_{i j}\right\|$, 得到
$$
\left|s_{i j}-s\right| \leq \tau_{i j}（7）
$$
 $s_{i j}=\left(\left\|\vec{y}_{i j}\right\|\right) /\left(\left\|\vec{x}_{i j}\right\|\right)$ 是两个向量比例 $\vec{x}_{i j}$ ， $\vec{y}_{i j}$,i和j对应点的边界： $\tau_{i j}=(2 \tau) /\left(\left\|\vec{x}_{i j}\right\|\right) .$  (7) 是缩放变量.  $s$ 是唯一未知量.$ \tau $表示误差限制值。

## Scale Estimation

 (7) 等价于求最大序列值, 即求取最大序列时的s：
$$
\begin{array}{l}
\max _{s, l^{s} \subseteq H}\left|I^{s}\right| \\
\text { s.t. } \frac{\left|s_{k}-s\right|}{\tau_{k}} \leq 1 \quad \forall k \in I^{s}
\end{array}
$$
$k$ 代替 $i j, \mathcal{H}=$ $\{1,2, \ldots, K\}$ 是观测s的下标$\left\{s_{k}\right\}_{1}^{K}$, $\left\{\tau_{k}\right\}_{1}^{K}$, $ \tau $是误差限制, 子集 $I^{s}$ 内值数,  $\left|I^{s}\right|$ 真值数. 最优缩放s对应的真值 $\tilde{I}^{s}$. 对$s_{k}$求最小二乘值, 
$$
\tilde{s}=\underset{s}{\arg \min } \sum_{k \in \tilde{I}^{*}}\left(\frac{s_{k}-s}{\tau_{k}}\right)^{2} .
$$
实际就是一个s的二次函数
$$
\tilde{s}=\left(\sum_{k \in \tilde{I}^{*}} \frac{1}{\tau_{k}^{2}}\right)^{-1} \sum_{k \in \tilde{I}^{s}} \frac{s_{k}}{\tau_{k}^{2}}（10）
$$
$N$ 个点对，有$K=$ $(N(N-1)) /(2)$向量. 复杂度为$O\left(K^{2}\right)=O\left(N^{4}\right)$. 提出one-point RANSAC减少复杂度。

![](/ransac_S求解.png)



## Rotation Estimation  

最小化目标函数

![](/r_esmate.png)

其中$ \rho $
$$
\rho\left(r_{k}, u\right)=\left\{\begin{array}{ll}
\frac{u^{2}}{6}\left(1-\left(1-\frac{r_{k}^{2}}{u^{2}}\right)^{3}\right), & \left|r_{k}\right| \leq u \\
\frac{u^{2}}{6}, & \left|r_{k}\right|>u
\end{array}\right.
$$
u是规模参数，控制规模大小:
$$
\underset{\mathbf{R}}{\operatorname{minimize}} \sum_{\left(\bar{x}_{k}, \bar{y}_{k}\right) \in \tilde{I}^{s}} w\left(r_{k}\right)\left\|\overrightarrow{\boldsymbol{y}}_{k}-\tilde{s} \mathbf{R} \overrightarrow{\boldsymbol{x}}_{k}\right\|^{2}（13）
$$
$w\left(r_{k}\right)=(\partial \rho) /\left(\partial r_{k}\right) / r_{k}$ 
$$
w\left(r_{k}, u\right)=\left\{\begin{array}{ll}
\left(1-\frac{r_{k}^{2}}{u^{2}}\right)^{2}, & \left|r_{k}\right| \leq u \\
0, & \left|r_{k}\right|>u（14）
\end{array}\right.
$$
函数图像见下图2：u越大对r的容忍度就越大

<img src="/Point Cloud Registration Based on One-Point——权重rou.jpg" style="zoom:600%;" />



算法流程：

![](/t_esmate.png)

## Translation Estimation  

将前两步求出的r，s求出后。将配准点对投影到最开始的点集，统计频率，越高就准确率越高，取百分之70.因为r和s已经求出，所以可以这么作

![](/t求解约束.png)



# the end：代码学习总结

## 数据读取问题

### 	先通过matlab查看数据结构

![维度](维度.png)

### 在python中加载数据

~~~python
from scipy import io as sio
 dataset = sio.loadmat(path)
#dataset是字典形式
#上图是他的键
#读取键，debug查看具体的结构
model = dataset.get('model')


~~~

### 寻找最近点对（注意scilearn 官方文档使用）

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





