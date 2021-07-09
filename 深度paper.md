---
title: paper
categories:
  - 论文
tags:
  - 深度
  - pytorch
date: 2021-03-07 20:48:16
description: <read more ...>
mathjax: true
---



# NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences

## 相关工作

### 参数方法（生成验证式）

​			：假设一个变换矩阵，用它来估计全局变换

​				缺点：1，当初始内联比低时，估计的全局变换的精度严重下降 2，非刚体变换和多视角

### 非参数方法（提取局部信息）：

​			例如：空间knn方法

### 学习式方法;

​			pointnet(点单独传入)，point net++（利用了部分局部性特征）

### 本文方法

​			Different from these learning-based methods for irregular data, our approach covers both locality selection and locality integration concerns via a compatibility metric and a hierarchical manner, respectively.



##  本文工作

### the importance of mining neighbors s

​		first，it explores the local space of each correspondence and ex-tracts local information by our proposed compatibility metric. 

​	    Second, it integrates unordered correspondences into a graph in which nodes correspond to the mined neighbors so that convolutions can be performed for further feature ex-
traction and aggregation.

### 算法步骤

​		Hessian-affine to find point

​		




# DCP: Deep Closest Point

## 1\. Registration的目标

对于两个本质“形状”接近的点云： ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-277b4891c53bdc722638e8e9dcb2bede.svg) ，如何通过旋转、平移，来调整 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) ，使 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) 尽量与 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) 重合。

## 2\. Registration的难点

点云中的点是无序的。旋转、平移，本质是对点云中每个点的调整。那么问题来了，**如何确定 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-277b4891c53bdc722638e8e9dcb2bede.svg) 中点的对应关系？**

传统的方法（ICP、Go-ICP等）提供的一些方案，其存在一些问题：很容易陷入局部最优点、太耗时。

论文提出了一种end-to-end **DCP算法（Deep Closest Point）**来处理Point Cloud Registration问题，相比之前的方法，性能提升很大。

## 3\. Transformer回顾

作者提出的DCP算法，里面使用了NIPS 2017 里面《[Attention Is All You Need](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)》中提出的Transformer网络结构。理解DCP算法之前，有必要对Transformer有一个大概的了解。

在seq2seq问题里面（例如机器翻译），基于RNN、LSTM的Encoder-Decoder方案主要的问题在于：

*   处理太长的sequence时，效果不佳；
*   RNN、LSTM后一步的输入，依赖于前一步的输出，因此很难实现并行化计算。

Transformer里面，并没有使用RNN等循环网络，而是使用一种self-attention的机制来实现seq2seq任务，在处理以上两个问题时，性能高于传统RNN方式的网络结构。

所以，**Transformer主要是一种基于self-attention的序列模型**。在处理Point Cloud Registration问题时，可以将两个待匹配的点云，分别看作两个序列（无序），那么，**Registration问题正好可以类比为一种特殊形式的seq2seq问题**，目标是找到两个序列（点云的点集合）之间的位置转换关系。作者对Transformer进行了一些修改后，用到了DCP网络中。

## 4\. 网络结构

![](D:\git_rep\hexo\source\_posts\dcp\1622984274-c409f6e011c8167c9987c833ee30741f.jpg)

  

整个网络分为：Initial Features、Attention、Pointer Generation、SVD Module四步。

**4.1 Initial Features**

使用一个共享的DGCNN网络分别计算点云![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-c731c61e4db30fe277feae2d2d9dcd96.svg) 中每个点的feature，得到：

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-609d2f97d086e3aa296ed99f3e6e5468.svg)

**4.2 Attention**

上面计算出来的 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-28103a1bd3ae33c23cae4479974c3664.svg) ，在计算过程中彼此独立，相互之间没有关系。为了让二者能产生关系，彼此知道对方的存在，从而更好的计算出“如何才能调整到对方的位置“，作者在这里使用了修改过的Transformer网络：

![](D:\git_rep\hexo\source\_posts\dcp\1622984274-ee0f2fb4cc2ec5b2d11de90461246a01.jpg)

关于该网络的细节，可以参考论文：[Attention Is All You Need](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)。（<strong style="color:red;">在这篇文章基础上改动的，特征提取网络也是作者自己写的</strong>）

Transformer的输出为：

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-3f38b8862d04e75052fe8176b6fecc09.svg)

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-a84b1e00ae490f5dbf83f19fced51508.svg) 相当于Transformer对feature学习到一个“条件”残差变化量。例如，对于 <img src="D:\git_rep\hexo\source\_posts\dcp\1622984274-3617a6b36361f59948a10177e31d5c4a.svg" alt="[公式]" style="zoom:50%;" /> ， ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-31799e50d252f85e8ae6b0f42fe2c003.svg) 表示要将 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\  1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) 调整到 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) ，应该怎么调整 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) 的feature。相当于以![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-879d55cfa66fc21c94c02718d5a6d59b.svg) 作为条件（condition），计算出 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-3617a6b36361f59948a10177e31d5c4a.svg) 的残差变化量。

至于为何不直接让 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-73193c0950cb5d7cf396f6b2110a46dc.svg)，而是先计算残差，再算出 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-2a541b4c61150e11214f6619f7bd2fe8.svg) ？可能实验表明这种方式性能会更好一些。可以自己做实验来验证。

## 4.3 Pointer Generation

前面说到，Registration的难点在于如何寻找两个点与点之间的对应关系。作者用 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-a48eddaf286db4872158974ed318daef.svg) 计算出soft pointer（类似于soft attention，soft map），得出一种基于概率的点与点的对应关系。具体做法如下：

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-f8af6066202be014a545014d3b65a8eb.svg) 表示点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) 经过Attention模块处理后， ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-390824601cf6d50933744af0e4ccf208.svg) 个点的feature；

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-2e23416afee0a9fd25a075f8966a7f5b.svg) 表示点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) 经过Attention模块处理后，其中的点 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 的feature；

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-7dc446e74a05c1dabb096d994b62403d.svg) 表示用点 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 的feature，分别与 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-fd3c128aef8aadfdf6ab67cc8e5eb460.svg) 中每个点的feature做点积，得到 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 与点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) 中每个点的相似度。（用点积计算相似度）；

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-1fcb8c047623145c678836623ec9a240.svg) 表示用 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-22158e2e6be1860c96524ad8b1436a2a.svg) 将相似度转化为概率，即 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 与点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) 中每个点的带权重的map关系。论文中称之为 _soft pointer、soft matching_。

**4.4 SVD Module**

得到点与点的对应关系后，下一步就是计算出到底该如何调整 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) ，使其尽量与 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) 对齐。

对于点 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-55abd6ee2c07065bfd8875b4e79a1cf9.svg) ，使用上面计算出来的与点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) 中每个点匹配的概率，加权求和，计算出来一个平均点：

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-499842f8e348e5dcb121fbd3112691a1.svg)

其中， ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-a58eef5e90227bfc61bdd11835af4d81.svg) 表示点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) 中N个点的坐标。 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-ff3b399485568c1abcd79f34d4ebcf3f.svg) 就是 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 调整的目标点。按相同的方法，可以计算出点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) 中每个点的目标点。最后使用SVD得出 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\ 1622984274-81651914e72da52a52f3baf300379792.svg) 的旋转、平移矩阵： ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-04e729b6b46ae88ef4a7c80ada1f5615.svg) 。

**4.5 Loss**

整个网络相当于输入点云 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-bb713738c1c0c8c066224f6258d3ff94.svg) 、 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-5080b1a67cc21b4d7d0e758935e59275.svg) ，输出 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-04e729b6b46ae88ef4a7c80ada1f5615.svg) 。用 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-04e729b6b46ae88ef4a7c80ada1f5615.svg) 与ground truth值来构建Loss：

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-54d733a26a690e190d1f020006f710d7.svg)

Loss中的第一项：因为希望 ![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-e645338ac2b74527c2f0e18e15138f50.svg) 是单位正交矩阵（Orthogonal Matrix），所以它的转置和逆应该相同。因此，理想情况下，以下等式应该成立：

![[公式]](D:\git_rep\hexo\source\_posts\dcp\1622984274-654ceab18a1a115b4d3561c77be25329.svg)

## 5\. 总结

论文将Transformer应用到了点云Registration问题中，通过Transformer中的attention机制，计算出一个“假想的目标点云“，这个**假想的目标点云**与**待调整点云**之间点的对应关系已知（soft matching）。再通过Loss约束，间接使得**假想的目标点云**向**真正的目标点云**不断逼近，最终实现点云的对齐匹配。



