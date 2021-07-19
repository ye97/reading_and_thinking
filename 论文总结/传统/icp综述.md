---
title: icp综述
categories:
  - 论文
  - 总结
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



**Iterative Closest Point** (**ICP**)

 [1](https://en.wikipedia.org/wiki/Iterative_closest_point#cite_note-1)  [2   ](https://en.wikipedia.org/wiki/Iterative_closest_point#cite_note-2)[3](https://en.wikipedia.org/wiki/Iterative_closest_point#cite_note-zhang_IJCV_1994-3) is an algorithm employed to minimize the difference between two clouds of points.

点云匹配分类法（1）

•全局匹配算法 Globe

•局部匹配算法Local

_Salvi, J. (2007). "A review of recent range image registration methods with accuracy evaluation." Image and Vision Computing 25(5): 578-596._  _Mellado, N. and D. Aiger (2014). "SUPER 4PCS Fast Global Point cloud Registration via Smart Indexing."_

点云匹配分类法（2）

•基于点的匹配

•基于特征的匹配

•点特征

•VPF

•FHPF

•…

•基于线特征

•"Algorithms for Matching 3D Line Sets."

•"Line segment-based approach for accuracy assessment of MLS point clouds in urban areas.“

•Poreba, M. and F. Goulette (2015). "A robust linear feature-based procedure for automated registration of point clouds." Sensors (Basel) 15(1): 1435-1457.

Coarse to fine registration粗-精过程

粗配的目的：提供刚体变换初始估计

Salvi, J., et al. (2007). 

改进ICP算法

_Besl, P. J. and N. D. Mckay (1992). "A Method for Registration of 3-D Shapes." IEEE Transactions on Pattern Analysis and Machine Intelligence 14(2): 239-256._  
_Siegwart, R., et al. (2015). "A Review of Point Cloud Registration Algorithms for Mobile Robotics." Foundations and Trends in Robotics._

•加快搜索效率

•K-D树

•Voronoi图

•不同的距离量测方式

•点到点

•点到线 PLICP

•Censi, A. (2008). "An ICP variant using a point-to-line metric." IEEE International Conference on Robotics & Automation. IEEE,: 19-25.

•CSM（Canonical Scan Matcher）源码     [http](http://censi.mit.edu/software/csm/)[://censi.mit.edu/software/csm](http://censi.mit.edu/software/csm/)[/](http://censi.mit.edu/software/csm/)

•点到面

•Low, K.-L. (2004).   

•面到面 GICP

ICP算法求解

•Closed Form

•SVD

•Unit Quaternions单位四元数

•The ICP error function minimization via orthonormal matrices

•Dual Quaternions

•数值解法

•LM算法 （Levenberg-Marquardt algorithm）

•Jerbić, B., et al. (2015). "Robot Assisted 3D Point Cloud Object Registration." Procedia Engineering 100: 847-852.

•点到面 线性最小二乘法

•Low, K.-L. (2004). "Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration."

问题

•观测误差

•部分重叠

•离群点Outlier、噪声（经常是错误点或者异常点）

•不满足一一对应的条件

解决方法

•剔除 Rejection

•PCL类库中采用

•权重方法

•稳健方法

_Bergström, P. and O. Edlund (2014). "Robust registration of point sets using iteratively reweighted least squares."_  
_H. Pottmann, S. Leopoldseder, and M. Hofer. Simultaneous registration of multiple views of a 3D object. ISPRS Archives 34/3A (2002), 265-270._  
_Andreas Nüchter(2008).3D Robotic Mapping-The Simultaneous Localization and Mapping Problem with Six Degrees of Freedom_

* * *

### 标准ICP

标准ICP算法是最早提出的基于点-点距离的算法，另外一种是基于点-面的算法，由chen提出，好多文献所说的恶Chen's Method。

标准的ICP算法需要粗配，满足距离足够近这一条件之后才能进行精确配准。

### IDC

The _idc_ algorithm does a point-to-point correspondence for calculating the scan alignment. The correspondence problem is solved by two heuristics: the closest point rule and the matching range rule. Furthermore, a formula is provided for calculating an error covariance matrix of the scan matching

**Trimmed ICP** 

在每次迭代的过程中，根据距离残差排序，按照重叠率计算保留的点数。根据保留的点进行计算变换。该方法可以很好的处理部分重叠问题。CC中采用该方法实现，作者的原文还提到了一种自适应计算重叠率的方法。推荐！

Chetverikov, D., et al., The Trimmed Iterative Closest Point algorithm. 2002. 3: p. 545-548.

**稳健ICP**

由于Outliner的存在，即观测误差和离群点存在，以及部分重叠问题，粗配之后的数据再进行精配的过程中仍然存在不稳健的问题（Robust问题），因此提出了稳健ICP方法。如SICP，IRLSICP

**MBICP**

**GICP 泛化的ICP，或者叫Plane to Plane ICP**

**EM-ICP**

**NICP**

**GO-ICP**

...

一般的ICP算法（上述的）是局部优化算法，还存在全局优化的问题，即不需要单独粗配，直接一步到位。很多的ICP算法都是稳健的方法，但是并不是全局的优化方法。全局的方法有Super4PCS、三点Ransac等。

[http://www.mathworks.com/matlabcentral/fileexchange/12627-iterative-closest-point-method](http://www.mathworks.com/matlabcentral/fileexchange/12627-iterative-closest-point-method)

[http://www.mathworks.com/matlabcentral/fileexchange/27804-iterative-closest-point](http://www.mathworks.com/matlabcentral/fileexchange/27804-iterative-closest-point)

[http://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration](http://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration)