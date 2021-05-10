

title: icp算法证明
categories:

  - 论文

date: 2021-03-13 10:16:46
description: <read more ...>
mathjax: true
typora-root-url:  icp算法证明

# ICP算法证明

## **问题引入**

        迭代最近点（Iterative Closest Point, 下简称ICP）算法是一种点云匹配算法。

        假设我们通过RGB-D相机得到了第一组点云 ![P = \left \{p _{1} ,p _{2} ,p _{3} ,\cdots ,p _{n} \right \}](icp证明/1620608087-7b6d2e32bd13e8873a5916ffd41a647f.latex)，相机经过位姿变换（旋转加平移）后又拍摄了第二组点云![Q = \left \{q _{1} ,q _{2} ,q _{3} ,\cdots ,q _{n} \right \}](icp证明/1620608087-686130d166c67964f8d2281dce493688.latex) ,注意这里的 ![P](icp证明/1620608087-6325ac7a6a3ea0b7773f7bac03a8f653.latex) 和 ![Q](icp证明/1620608087-91d4657f758d80796061a4d7d476ba60.latex) 的坐标分别对应移动前和移动后的坐标系（即坐标原点始终为相机光心，这里我们有移动前、移动后两个坐标系），并且我们通过相关算法筛选和调整了点云存储的顺序，使得 ![P](icp证明/1620608087-6325ac7a6a3ea0b7773f7bac03a8f653.latex)和![Q](icp证明/1620608087-91d4657f758d80796061a4d7d476ba60.latex) 中的点一一对应，如![\left (P _{99},Q _{99} \right )](icp证明/1620608087-5ef27db860d1f7822b26cb9aca3a4bb6.latex)在三维空间中对应同一个点。

        现在我们要解决的问题是：计算相机的旋转  ![R](icp证明/1620608087-ddfe2a175e23e49d70fd49f77c4c8221.latex) 和平移 ![t](icp证明/1620608087-b5eeaedf6ad956f27dfd8421ea1c8e45.latex) ，在没有误差的情况下，从![P](icp证明/1620608087-6325ac7a6a3ea0b7773f7bac03a8f653.latex) 坐标系转换到![Q](icp证明/1620608087-91d4657f758d80796061a4d7d476ba60.latex) 的公式为：

                                              ![q_{i} = Rp_{i} + t](icp证明/1620608087-34738130a734b48eccef5c3f7c970e10.latex)

        但由于噪声及错误匹配（如 ![\left (P _{99},Q _{99} \right )](icp证明/1620608087-5ef27db860d1f7822b26cb9aca3a4bb6.latex) 其实并不对应空间中同一点，但特征匹配算法错误地认为二者是同一点）的存在， 上式不总是成立，所以我们要最小化的目标函数为

                                             ![\frac{1}{2}\sum_{i = 1}^{n}\left \| q_{i} - Rp_{i} - t \right \|^{2}](icp证明/1620608087-20f16dbe751a2c3239f7ec5202719ef1.latex)

        常用的求解 ![R](icp证明/1620608087-ddfe2a175e23e49d70fd49f77c4c8221.latex) 和 ![t](icp证明/1620608087-b5eeaedf6ad956f27dfd8421ea1c8e45.latex) 的方法有：

1.  SVD
2.  非线性优化

        非线性优化的描述比较繁琐，下面只介绍SVD方法。为了后面能使用SVD，我们需要在数学上做一点小小的变形。首先定义两组点云的质心（center of mass）为 ![\mu _{p} = \frac{1}{n} \sum_{i = 1}^{n}p_{i}](icp证明/1620608087-9f248482c67c244b2fc9e39472d608c8.latex),![\mu _{q} = \frac{1}{n} \sum_{i = 1}^{n}q_{i}](icp证明/1620608087-eae2041b5e83420896ec3aa6b677ac51.latex),并作出如下处理：

![](icp证明/1620608087-36ae0bb46b8b434797ba6bce09d9664e.png)

![](icp证明/1620608087-ab16695bc54b37f5da0d891ab02673ed.png)

此处r等于UV参见论文，简单得说就是保证秩最大。

此处要记住得就是R的求解定式。

