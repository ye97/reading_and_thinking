---
title: paper
categories:
  - 论文
tags:
  - 深度
  - pytorch
date: 2021-03-07 20:48:16
mathjax: true
---



# NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences

## 1.1 相关工作

### 1.1.1 参数方法（生成验证式）

​			：假设一个变换矩阵，用它来估计全局变换

​				缺点：1，当初始内联比低时，估计的全局变换的精度严重下降 2，非刚体变换和多视角

### 1.1.2 非参数方法（提取局部信息）：

​			例如：空间knn方法

### 1.1.3 学习式方法;

​			point net(点单独传入)，point net++（利用了部分局部性特征）

### 1.1.4 本文方法

​			Different from these learning-based methods for irregular data, our approach covers both locality selection and locality integration concerns via a compatibility metric and a hierarchical manner, respectively.



## 1.2 本文工作

### 1.2.1，the importance of mining neighbors s

​		first，it explores the local space of each correspondence and ex-
tracts local information by our proposed compatibility metric. 

​	    Second, it integrates unordered correspondences into a graph in which nodes correspond to the mined neighbors so
that convolutions can be performed for further feature ex-
traction and aggregation.

### 1.2.2 算法步骤

​		Hessian-affine to find point

​		

​								

