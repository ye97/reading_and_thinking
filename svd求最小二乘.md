---
title: svd求最小二乘
categories:
  - 论文
  - 传统点云
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-05-19 10:45:24
urlname:
tags:
typora-root-url: svd求最小二乘
---

# 使用奇异值分解的最小二乘刚性配准方法


1、问题描述

![\mathbb{P} = \left \{ p_1,p_2,...,p_n \right \}](1621392256-5e8dce9c7688d714c0eb8b8defcf15a8.latex)

![\mathbb{Q} = \left \{ q_1,q_2,...,q_n \right \}](1621392256-7d17b469d262dc119f3d4a4a6c1631fb.latex)

是变换空间![\mathbb{R}^d](1621392256-c83d1c9c147b3fbece79fe6f67a6d644.latex)中对应的两组点集。

我们希望得到一个刚体变换是的两组点集最小二乘结果最小

![\left (1621392256-721944295b50592cd6f51d0d84854cd1.latex ) = argmin_{R\in SO\left ( d \right ), t\in \mathbb{R}^d}](1621392256-721944295b50592cd6f51d0d84854cd1.latex)![\left (1621392256-5c88fad5d9bd256018a421f7a0e33a2e.latex )=\mathop{\arg\min}_{R\in SO\left ( d \right ), t\in \mathbb{R}^d)} \sum_{i=1}^{n}w_i \left \| \left ( R p_i + t\right ) - q_i\right \|^2](1621392256-5c88fad5d9bd256018a421f7a0e33a2e.latex)  ---------------------(1)

其中R就是旋转矩阵，t是平移向量，w\_i就是每对点的权重。

2、计算平移

首先假定R是固定的，求解公式就是![F\left (1621392256-a7e24a42c49bf34dba40b409e3cd7ddb.latex )= \sum_{i=1}^{n}w_i \left \| \left ( R p_i + t\right ) - q_i\right \|^2](1621392256-a7e24a42c49bf34dba40b409e3cd7ddb.latex)通过对F(t)求导，t的最优解就可以知道了

![0 = \frac{\partial F}{\partial t} =\sum_{i=1}^{n}2 w_i\left (1621392256-12a8af81acd55c294f8ef1c9bb12e09e.latex ) =2t\left ( \sum_{i=1}^{n}w_i \right ) + 2R\left ( \sum_{i=1}^{n}w_i p_i \right )-2\left ( \sum_{i=1}^{n}w_i q_i \right )](1621392256-12a8af81acd55c294f8ef1c9bb12e09e.latex)\--------------(2)

左边是0，那我们直接将除以![\sum_{i=1}^{n}w_i](1621392256-5679cfbe30ce30b765d2a1f8feb5cc31.latex)

![\overline{p} = \frac{\sum_{i=1}^{n}w_ip_i}{\sum_{i=1}^{n}w_i}](1621392256-2cb9dac1e60d0e654704b942b05a9511.latex)、![\overline{q} = \frac{\sum_{i=1}^{n}w_iq_i}{\sum_{i=1}^{n}w_i}](1621392256-85efc55992af14079895255a725cd36e.latex)\--------------------------------------------------------------------------------(3)

公式（2）就整理成了

![t = \overline{q}-R\overline{p}](1621392256-31a3df4d9a97772c9fff40739cbdc9f7.latex)\--------------------------------------------------------------------------------------------------------------(4)

最优平移t将转换后的P的加权质心映射到Q的加权质心

我们在把公式（4）带入目标函数F：

![F\left (1621392256-6a474202f025776d256756cfbb17dfe2.latex )= \sum_{i=1}^{n}w_i \left \| \left ( R p_i +\overline{q}-R\overline{p}\right ) - q_i\right \|^2 = \sum_{i=1}^{n}w_i \left \| R \left ( p_i -\overline{p}\right ) -\left ( q_i-\overline{q} \right )\right \|^2](1621392256-6a474202f025776d256756cfbb17dfe2.latex)\---------------------（6）

我们再次引入新的定义

![x_i :=p_i - \overline{p}, y_i:=q_i-\overline{q}](1621392256-93a77f407e640f367ec4ad9e002d032b.latex)\------------------------------------------------------------------------------------------(7)

那么最优的旋转求解就可以变成

![R=\mathop{\arg\min}_{R\in SO\left (1621392256-d783fbaa0659a7a732e04f4eb43f18d1.latex )} \sum_{i=1}^{n}w_i \left \| R x_i - y_i\right \|^2](1621392256-d783fbaa0659a7a732e04f4eb43f18d1.latex)\-------------------------------------------------------------------------------(8)

3 计算旋转矩阵

将公式（8）展开

![\left \| R x_i - y_i\right \|^2 = \left (1621392256-84e2010f11d0dccc0e1fa0e7b0dbecca.latex )^T\left ( Rx_i-y_i \right ) =\left ( x_i^TR^T-y_i^T \right )\left ( Rx_i-y_i \right ) =x_i^TR^TRx_i-y_i^TRx_i -x_i^TR^Ty_i+y_i^Ty_i =x_i^Tx_i-y_i^TRx_i -x_i^TR^Ty_i+y_i^Ty_i](1621392256-84e2010f11d0dccc0e1fa0e7b0dbecca.latex)\-----（9）

旋转矩阵![R^TR=I](1621392256-39b7825b922d0bd8560c502c80bcd004.latex)   推到的话  ![\left (1621392256-beab6046d526a0795866903b2b69ed79.latex )^T\left ( R_zR_yR_x \right )](1621392256-beab6046d526a0795866903b2b69ed79.latex)比较容易了![R_z^T R_z=I](1621392256-8a5e73e109135128b318addddb4dd650.latex)

![x_i^TR^Ty_i](1621392256-dbb8345af82097d1689512916bd8f7c0.latex)是一个标量，证明过程主要是根据维度来的![x_i^T](1621392256-6aaf8eea8f4962af1396ab8921440553.latex)是1Xd,![R^T](1621392256-751a3ca6913018ed0474cc804f4d538b.latex)是dxd，![y_i](1621392256-8fd61711876411734a9a1330f41c4880.latex)是dx1,可不就是标量么。标量的转置可不就是标量本身么

![x_i^TR^Ty_i = \left (1621392256-b7e48c96c8d1639af510646c3f96f722.latex )^T = y_i^TRx_i](1621392256-b7e48c96c8d1639af510646c3f96f722.latex)\---------------------------------------------------------------------------------(10)

因此公式（9）就可以展开为

![\left \| R x_i - y_i\right \|^2 =x_i^Tx_i-2y_i^TRx_i +y_i^Ty_i](1621392256-be46c538fef0355bbf12148cce442132.latex)\-------------------------------------------------------------------------(11)

然后我们将公式（11）带入到公式（8）

![R=\mathop{\arg\min}_{R\in SO\left (1621392256-c3ad0650132fe908e9c0a6567d1c9b33.latex )} \sum_{i=1}^{n}w_i \left \| R x_i - y_i\right \|^2 =\mathop{\arg\min}_{R\in SO\left ( d \right )} \sum_{i=1}^{n}w_i \left ( x_i^Tx_i-2y_i^TRx_i +y_i^Ty_i \right ) =\mathop{\arg\min}_{R\in SO\left ( d \right )} \left ( \sum_{i=1}^{n}w_i x_i^Tx_i-2\sum_{i=1}^{n}w_i y_i^TRx_i +\sum_{i=1}^{n}w_i y_i^Ty_i \right ) =\mathop{\arg\min}_{R\in SO\left ( d \right )} \left (-2\sum_{i=1}^{n}w_i y_i^TRx_i \right )](1621392256-c3ad0650132fe908e9c0a6567d1c9b33.latex)\----（12）

标量在求解最小化中没啥用啊，可以删掉，还有标量2也可以删掉，因此公式（12）最终可以表示为

![\mathop{\arg\min}_{R\in SO\left (1621392256-73c6e666543c5951afa5c20e7ebaaa1d.latex )} \left (-2\sum_{i=1}^{n}w_i y_i^TRx_i \right )=\mathop{\arg\max}_{R\in SO\left ( d \right )} \left (\sum_{i=1}^{n}w_i y_i^TRx_i \right )](1621392256-73c6e666543c5951afa5c20e7ebaaa1d.latex)\-----------------------------------------------（13）

我们在把公式（13）中的![\sum_{i=1}^{n}w_i y_i^TRx_i = tr\left (1621392256-12a5fdbffc6acf6e647ad042873408ba.latex )](1621392256-12a5fdbffc6acf6e647ad042873408ba.latex)进行矩阵表示

![\begin{bmatrix} w_1 & & & \\ & w_2 & & \\ & & \ddots & & \\ & & & w_n \end{bmatrix}\begin{bmatrix} - y_1^T - \\ - y_2^T - \\ - \vdots - \\ - y_n^T - \end{bmatrix}\begin{bmatrix} & & & \\ & R & \\ & & & \end{bmatrix}\begin{bmatrix} | | | | \\ x_1 x_2 \cdots x_n \\ |||| \end{bmatrix} =\begin{bmatrix} - w_1y_1^T - \\ - w_2y_2^T - \\ - \vdots - \\ - w_ny_n^T - \end{bmatrix}\begin{bmatrix} | | | | \\ Rx_1 Rx_2 \cdots Rx_n \\ |||| \end{bmatrix} =\begin{bmatrix} w_1y_1^TRx_1 & & & \\ & w_2y_2^TRx_2 & & \\ & & \ddots & & \\ & & & w_ny_n^TRx_n \end{bmatrix}](1621392256-266541c01e1b7c1c0286178a648ab3f9.latex)

![W=diag\left (1621392256-0fd024c14d2c22fc537205ff7114475c.latex )](1621392256-0fd024c14d2c22fc537205ff7114475c.latex)是一个nxn对角矩阵，权重w\_i是第i个对角元素。Y是dxn矩阵y\_i是列向量，X是dxn矩阵x\_i是列向量。方阵的迹是对角线上元素的和。

![\sum_{i=1}^{n}w_i y_i^TRx_i = tr\left (1621392256-12a5fdbffc6acf6e647ad042873408ba.latex )](1621392256-12a5fdbffc6acf6e647ad042873408ba.latex)\-----------------------------------------------------------------------------------------(14)

求解旋转矩阵就是最大化矩阵的迹![tr\left (1621392256-593a1d25cc706f275086a42948f18d1f.latex )](1621392256-593a1d25cc706f275086a42948f18d1f.latex)

矩阵的迹可符合交换律,前提是AB的维度要可以相乘啊

![tr\left (1621392256-a653908f07d8aa5102784659097ec2f7.latex )=tr\left ( BA\right )](1621392256-a653908f07d8aa5102784659097ec2f7.latex)\-------------------------------------------------------------------------------------------------------（15）

![tr\left (1621392256-fcdb547f644b809aeae765245156a7ec.latex )=tr\left ( \left ( WY^T \right ) \left ( RX \right )\right )=tr\left ( \left ( RX \right )\left ( WY^T \right ) \right )=tr\left ( RX WY^T \right )](1621392256-fcdb547f644b809aeae765245156a7ec.latex)\-----------------------（16）

看矩阵维度X就是dxn，W是nxn，Y^T是nxd

问题来了![S=XWY^T](1621392256-45acfd110a0dc0a611b6fc196df0ed5e.latex)是一个dxd的矩阵，这么一看是不是像一个协方差矩阵“covariance”（我也不知道为什么像）

我们现在将S进行SVD：

![S=U\Sigma V^T](1621392256-bad0451c1e9b20c54b7027a5a388ef7a.latex)\-------------------------------------------------------------------------------------------------------------------(17)

在将公式（17）带入公式（16）

![tr\left (1621392256-d6af3458cf14e6904f6b48f0ae8d9a6f.latex )=tr\left ( RS \right )=tr\left ( RU\Sigma V^T \right )=tr\left ( \Sigma V^TRU \right )](1621392256-d6af3458cf14e6904f6b48f0ae8d9a6f.latex)\--------------------------------------------------(18)

其实V,U是正交矩阵，这个好理解R也是正交矩阵，旋转矩阵就是正交矩阵啊，所以![M=V^TRU](1621392256-42e19dcc25013f7506c887c31b17fa4a.latex)也是一个正交矩阵，也就是M的列是正交向量，![m_j^Tm_j=1](1621392256-68728db80178c2d7288102bdb7581d34.latex),m\_j是M的列向量，然后更好理解了m\_ij<=1

![1=m_j^Tm_j=\sum _{i=1}^{d}m_{ij}^{2}\Rightarrow m_{ij}^{2}\Rightarrow\left | m_ij \right |\leqslant 1](1621392256-f7f62e9e64e89c204f87d709297fc673.latex)\------------------------------------------------------------------------(19)

![tr\left (1621392256-67066f28e17fdb8c66bd13a27d99a291.latex )](1621392256-67066f28e17fdb8c66bd13a27d99a291.latex)最大可能是什么呢，![\Sigma](1621392256-a8b0c2500a188ba6d859410b959db675.latex)是非负特征值![\sigma_1,\sigma_1,\cdots ,\sigma_d\geq 0](1621392256-570e70153e8717ebee145c8f4cfe6bec.latex)为对角线元素的对角矩阵，因此

![tr\left (1621392256-552142174c917d139b4cf22c2ba4c10d.latex )=\begin{pmatrix} \sigma_1 & & & \\ & \sigma_2 & & \\ & & \ddots & \\ & & & \sigma_d \end{pmatrix}\begin{pmatrix} m_{11} m_{12} \cdots m_{1d} \\ m_{21} m_{22} \cdots m_{2d} \\ \vdots \vdots \vdots \vdots\\ m_{d1} m_{d2} \cdots m_{dd} \end{pmatrix} =\sum_{i=1}^{d}\sigma_i m_{ii}\leq \sum_{i=1}^{d}\sigma_i](1621392256-552142174c917d139b4cf22c2ba4c10d.latex)\---------------------------(20)

那这个![tr\left (1621392256-67066f28e17fdb8c66bd13a27d99a291.latex )](1621392256-67066f28e17fdb8c66bd13a27d99a291.latex)迹的最大值不就是当m\_ii=1的时候么，M是个正交矩阵，那M就只能是单位矩阵了

![I=M=V^TRU\Rightarrow V=RU\Rightarrow R=VU^T](1621392256-9cd87eb00515718abdef5902a25a3372.latex)\----------------------------------------------------------------------(21)

Orientation rectification，上面的描述都是如何找到最优的正交矩阵，但是旋转矩阵中也有反射变换。这些假设其实都是认为这些点集可以通过变换很好的对齐，但如果只有旋转变换，实际上也许一个都对不齐。

检查![R=VU^T](1621392256-b217334fc771defd1d054534a4fb409c.latex)是否为旋转矩阵，如果![det\left (1621392256-0a2a42ac64d26612c50cd6cd18692104.latex )=-1](1621392256-0a2a42ac64d26612c50cd6cd18692104.latex)这就包含了反射，否则![det\left (1621392256-cf3af215ac043daca501bbe162d1295a.latex )=1](1621392256-cf3af215ac043daca501bbe162d1295a.latex)。假设![det\left (1621392256-0a2a42ac64d26612c50cd6cd18692104.latex )=-1](1621392256-0a2a42ac64d26612c50cd6cd18692104.latex)，那么R是一个旋转矩阵等价于M是一个反射矩阵，我们现在想找到一个反射矩阵M![tr\left (1621392256-d4f7694019f19b793fff21e0ca359cf0.latex )=\sigma_1m_{11}+\sigma_2m_{22}+\cdots +\sigma_dm_{dd}=:f\left ( m_{11},\cdots ,m_{dd} \right )](1621392256-d4f7694019f19b793fff21e0ca359cf0.latex)\-----------------------------------------(22)

现在f只依赖于对角矩阵M，没有其他人变量，m\_ii看做是变量![\left (1621392256-6625868f842c8da30b40be0f55ea8512.latex )](1621392256-6625868f842c8da30b40be0f55ea8512.latex)，这是n阶反射矩阵的对角线的集合。n阶旋转矩阵的所有对角线集合等于值为-1的坐标个数为偶数的点集![\left (1621392256-89577cb32bf9f60366804b582475c7b2.latex )](1621392256-89577cb32bf9f60366804b582475c7b2.latex)的凸包，由于任何反射矩阵都可以通过翻转旋转矩阵的一行符号来构造，反之亦然，因此我们所优化的点集![\left (1621392256-89577cb32bf9f60366804b582475c7b2.latex )](1621392256-89577cb32bf9f60366804b582475c7b2.latex)凸包中-1的个数是uneven（我也不知道是个啥，应该是非偶数）。

由于定义域是一个凸多面体，线性函数f在顶点处达到极值。对角矩阵![\left (1621392256-7d0921912ee58c4355364f9b0df31a65.latex )](1621392256-7d0921912ee58c4355364f9b0df31a65.latex)因为它有偶数个- 1（也就是0了），所以不在定义域内，所以最简洁的形式就是![\left (1621392256-29ea05072984cda75d1bd36d094a41d6.latex )](1621392256-29ea05072984cda75d1bd36d094a41d6.latex)：

![tr\left (1621392256-aa94ea22faccc5b4dde6693b9339d456.latex )=\sigma_1+\sigma_2+\cdots +\sigma_{d-1}-\sigma_{d}](1621392256-aa94ea22faccc5b4dde6693b9339d456.latex)\-------------------------------------------------------------------------------（23）

这个值是在定义域的一个顶点得到的，并且大于除![\left (1621392256-b30fc5bb338ae805e0a3a080cf8288f7.latex )](1621392256-b30fc5bb338ae805e0a3a080cf8288f7.latex)任何形式的组合![\left (1621392256-89577cb32bf9f60366804b582475c7b2.latex )](1621392256-89577cb32bf9f60366804b582475c7b2.latex)，因为![\sigma_{d}](1621392256-6ce20b49a0c7241dd61d0ee30ea4610b.latex)是最小的特征值啊。

要记住我们一直是一个优化问题啊，

![M=V^TRU=\begin{pmatrix} 1 & & & & \\ & 1& & & &\\ & & \ddots & & &\\ & & & 1& & \\ & & & & 1 \end{pmatrix} \Rightarrow R = V\begin{pmatrix} 1 & & & & \\ & 1& & & &\\ & & \ddots & & &\\ & & & 1& & \\ & & & & 1 \end{pmatrix} U^T](1621392256-602e75708defb31c27d6ac4e49e600ec.latex)\----------------------------(24)

我们可以把有没有反射情况的通解写在一起，也就是![det\left (1621392256-cf3af215ac043daca501bbe162d1295a.latex )=1](1621392256-cf3af215ac043daca501bbe162d1295a.latex)、![det\left (1621392256-0a2a42ac64d26612c50cd6cd18692104.latex )=-1](1621392256-0a2a42ac64d26612c50cd6cd18692104.latex)

![R = V\begin{pmatrix} 1 & & & & \\ & 1& & & &\\ & & \ddots & & &\\ & & & 1& & \\ & & & & det\left (1621392256-a84a90d4fe62327e447283e012508369.latex ) \end{pmatrix} U^T](1621392256-a84a90d4fe62327e447283e012508369.latex)\-----------------------------------------------------------------------(25)

4、刚体变换求解总结

最优化求解平移t和旋转R等价于最小化

![\sum_{i=1}^n w_i\left \| \left (1621392256-e16e5885e9393f10821ac8c1c4d0e783.latex ) -q_i \right \|^2](1621392256-e16e5885e9393f10821ac8c1c4d0e783.latex)

1.计算两个点集的加权形心。

![\overline{p} = \frac{\sum_{i=1}^{n}w_ip_i}{\sum_{i=1}^{n}w_i}](1621392256-2cb9dac1e60d0e654704b942b05a9511.latex)、![\overline{q} = \frac{\sum_{i=1}^{n}w_iq_i}{\sum_{i=1}^{n}w_i}](1621392256-85efc55992af14079895255a725cd36e.latex)

2.计算所有点到中心的向量

![x_i :=p_i - \overline{p}, y_i:=q_i-\overline{q}](1621392256-93a77f407e640f367ec4ad9e002d032b.latex)

3计算dxd的协方差矩阵（啥协方差不协方差的？）

![S=XWY^T](1621392256-45acfd110a0dc0a611b6fc196df0ed5e.latex)

X、Y分别是dxn的矩阵分别是有x\_i和y\_i列向量组成，W就是对角矩阵了![W=diag\left (1621392256-0fd024c14d2c22fc537205ff7114475c.latex )](1621392256-0fd024c14d2c22fc537205ff7114475c.latex)

4.计算特征值分解![S=U\Sigma V^T](1621392256-bad0451c1e9b20c54b7027a5a388ef7a.latex)这时候旋转矩阵不就是

![R = V\begin{pmatrix} 1 & & & & \\ & 1& & & &\\ & & \ddots & & &\\ & & & 1& & \\ & & & & det\left (1621392256-a84a90d4fe62327e447283e012508369.latex ) \end{pmatrix} U^T](1621392256-a84a90d4fe62327e447283e012508369.latex)

5.计算平移

![t = \overline{q}-R\overline{p}](1621392256-31a3df4d9a97772c9fff40739cbdc9f7.latex)

注意啊，旋转中心是0,0,0