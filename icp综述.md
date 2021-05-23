---
title: icp综述
categories:
  - 论文
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-05-23 19:04:54
urlname:
tags:
---

* # ICP算法

  * * *

  ICP（Iterative Closest Point），即迭代最近点算法，是经典的数据配准算法。其特征在于，通过求取源点云和目标点云之间的对应点对，基于对应点对构造旋转平移矩阵，并利用所求矩阵，将源点云变换到目标点云的坐标系下，估计变换后源点云与目标点云的误差函数，若误差函数值大于阀值，则迭代进行上述运算直到满足给定的误差要求.

  ICP算法采用最小二乘估计计算变换矩阵，原理简单且具有较好的精度，但是由于采用了迭代计算，导致算法计算速度较慢，而且采用ICP进行配准计算时，其对配准点云的初始位置有一定要求，若所选初始位置不合理，则会导致算法陷入局部最优。

  ## Align 3D Data

  如果空间中两组点云之间的对应关系已经明确，则很容易求得两者之间的刚性变换，即旋转和平移共6个参数，但这种对应关系一般很难事先知道。  
  ICP算法假设两组点云之间的对应关系由最近点确定，一步步将源点云(P)匹配到目标点云(Q)。

  ICP算法主要包含对应点确定和变换计算更新，简要流程如下

  1.  在源点云 (P) 中选择一些随机点 (p\_i, i=1,2, .....,n)
  2.  在目标点云 (Q) 中找到每个点 ($p_i$) 的最近点 ($q_i$)
  3.  剔除一些距离较远的点对
  4.  构建距离误差函数(E)
  5.  极小化误差函数，如果对应点距离小于给定阈值设置，则算法结束；否则根据计算的旋转平移更新源点云，继续上述步骤。

  ## Basic ICP

  传统的ICP算法主要有两种度量函数，即point-to-point和point-to-plane距离，一般来说，point-to-plane距离能够加快收敛速度，也更加常用。

  ## Colored ICP

  Colored ICP算法_\[Park2017\]_针对有颜色的点云，在原始point-to-plane能量项的基础上，增加了一个对应点对之间的颜色约束，能够有更好的配准结果。

  ## Symmetrized ICP

  Symmetrized ICP算法极小化到点到由(n\_p)和(n\_q)决定的平面的距离平方和，同时优化拆分的旋转 :

  ## Symmetric ICP

  Symmetric ICP_\[Rusinkiewicz2019\]_是ICP算法的另一种改进。

  ## Solve

  ICP算法在极小化能量时通常都需要求解一个非线性最小二乘问题，但可以线性化，可以得到一个线性的最小二乘问题，再用Gauss-Newton或者Levenberg-Marquardt算法求解。

  ## Algorithm

  ## Go-ICP

  Go-ICP即Globally optimal ICP，提出了在L2误差度量下欧式空间中匹配两组点云的全局最优算法。

  ### Sparse ICP

  ## Code

  *   ICP: [libicp](https://github.com/symao/libicp) [Iterative-Closest-Point](https://github.com/Gregjksmith/Iterative-Closest-Point) [Go-ICP](https://github.com/yangjiaolong/Go-ICP)
  *   Sparse ICP: [sparseicp](https://github.com/OpenGP/sparseicp) [icpSparse](https://github.com/palanglois/icpSparse)
  *   CUDA
      *   [ICPCUDA](https://github.com/mp3guy/ICPCUDA)
      *   [CudaICP](https://github.com/akselsv/CudaICP)
      *   [ICP](https://github.com/FeeZhu/ICP)
      *   [pose\_refine](https://github.com/meiqua/pose_refine)
  *   OpenCL
      *   [ICP](https://github.com/nlamprian/ICP)

  ## Reference

  *   [Dynamic Geometry Processing](http://resources.mpi-inf.mpg.de/deformableShapeMatching/EG2012_Tutorial/slides/1.2%20ICP_+_TPS_%28NM%29.pdf)

  *   \[Rusinkiewicz2019\] Szymon Rusinkiewicz. A Symmetric Objective Function for ICP, SIGGRAOH 2019

  *   \[Park2017\] J. Park, Q.-Y. Zhou, and V. Koltun. Colored Point Cloud Registration Revisited, ICCV 2017.

  *   \[Bouaziz2013\] Sofien Bouaziz, Andrea Tagliasacchi, Mark Pauly. Sparse Iterative Closest Point, SGP 2013.

