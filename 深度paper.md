---
title: 深度paper
categories:
  - 论文
  - 深度
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



# dcp：deep cloest point

​		

##  Registration的目标

对于两个本质“形状”接近的点云： ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-277b4891c53bdc722638e8e9dcb2bede.svg) ，如何通过旋转、平移，来调整 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94.svg) ，使 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94-1625904790008.svg)尽量与 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) 重合。

## Registration的难点

点云中的点是无序的。旋转、平移，本质是对点云中每个点的调整。那么问题来了，**如何确定 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-277b4891c53bdc722638e8e9dcb2bede.svg) 中点的对应关系？**

传统的方法（ICP、Go-ICP等）提供的一些方案，其存在一些问题：很容易陷入局部最优点、太耗时。

现有方法有ICP以及其变种方法，即通过迭代优化点云对应关系和变换矩阵来得到最终的最优解。但是这种方法由于目标函数是非凸函数，倾向于陷入局部最小值，在点云初始对应距离过大时甚至会得到伪最优。加入启发式的ICP变种方法速度满，且精度没有很大提升。

此外还有基于深度学习的配准方法PointNetLK，精度也不是很高。

论文提出了一种end-to-end **DCP算法（Deep Closest Point）**来处理Point Cloud Registration问题，相比之前的方法，性能提升很大。

##  Transformer回顾

作者提出的DCP算法，里面使用了NIPS 2017 里面《[Attention Is All You Need](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)》中提出的Transformer网络结构。理解DCP算法之前，有必要对Transformer有一个大概的了解。

在seq2seq问题里面（例如机器翻译），基于RNN、LSTM的Encoder-Decoder方案主要的问题在于：

*   处理太长的sequence时，效果不佳；
*   RNN、LSTM后一步的输入，依赖于前一步的输出，因此很难实现并行化计算。

Transformer里面，并没有使用RNN等循环网络，而是使用一种self-attention的机制来实现seq2seq任务，在处理以上两个问题时，性能高于传统RNN方式的网络结构。

所以，**Transformer主要是一种基于self-attention的序列模型**。在处理Point Cloud Registration问题时，可以将两个待匹配的点云，分别看作两个序列（无序），那么，**Registration问题正好可以类比为一种特殊形式的seq2seq问题**，目标是找到两个序列（点云的点集合）之间的位置转换关系。作者对Transformer进行了一些修改后，用到了DCP网络中。

## 网络结构

![](%E6%B7%B1%E5%BA%A6paper/1625904000-c691129386195f45b9e85d86db6a6243.jpg)

  

整个网络分为：Initial Features、Attention、Pointer Generation、SVD Module四步。

**4.1 Initial Features**

使用一个共享的DGCNN网络分别计算点云![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-c731c61e4db30fe277feae2d2d9dcd96.svg) 中每个点的feature，得到：

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-609d2f97d086e3aa296ed99f3e6e5468.svg)

输入：点云X，Y

输出：特征$F_x$,$F_y$ 


这里作者选择了PointNet和DGCNN作为特征提取器。

其中DGRCNN相比于PointNet额外编码了局部几何信息，作者认为这将有助于配准精度。

**4.2 Attention**

上面计算出来的 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-28103a1bd3ae33c23cae4479974c3664.svg) ，在计算过程中彼此独立，相互之间没有关系。为了让二者能产生关系，彼此知道对方的存在，从而更好的计算出“如何才能调整到对方的位置“，作者在这里使用了修改过的Transformer网络：

![](%E6%B7%B1%E5%BA%A6paper/1625904000-6028d827984cb4f1be6a07227d0f86de.jpg)

关于该网络的细节，可以参考论文：[Attention Is All You Need](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)。

Transformer的输出为：

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-3f38b8862d04e75052fe8176b6fecc09.svg)

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-a84b1e00ae490f5dbf83f19fced51508.svg) 相当于Transformer对feature学习到一个“条件”残差变化量。例如，对于 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-3617a6b36361f59948a10177e31d5c4a.svg) ， ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-31799e50d252f85e8ae6b0f42fe2c003.svg) 表示要将 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94.svg) 调整到 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) ，应该怎么调整 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94.svg) 的feature。相当于以![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-879d55cfa66fc21c94c02718d5a6d59b.svg) 作为条件（condition），计算出 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-3617a6b36361f59948a10177e31d5c4a.svg) 的残差变化量。

至于为何不直接让 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-73193c0950cb5d7cf396f6b2110a46dc.svg)，而是先计算残差，再算出 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-2a541b4c61150e11214f6619f7bd2fe8.svg) ？可能实验表明这种方式性能会更好一些。可以自己做实验来验证。

##  Pointer Generation

前面说到，Registration的难点在于如何寻找两个点与点之间的对应关系。作者用 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-a48eddaf286db4872158974ed318daef.svg) 计算出soft pointer（类似于soft attention，soft map），得出一种基于概率的点与点的对应关系。具体做法如下：

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-f8af6066202be014a545014d3b65a8eb.svg) 表示点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) 经过Attention模块处理后， ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-390824601cf6d50933744af0e4ccf208.svg) 个点的feature；

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-2e23416afee0a9fd25a075f8966a7f5b.svg) 表示点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94.svg) 经过Attention模块处理后，其中的点 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 的feature；

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-7dc446e74a05c1dabb096d994b62403d.svg) 表示用点 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 的feature，分别与 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-fd3c128aef8aadfdf6ab67cc8e5eb460.svg) 中每个点的feature做点积，得到 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 与点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) 中每个点的相似度。（用点积计算相似度）；

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-1fcb8c047623145c678836623ec9a240.svg) 表示用 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-22158e2e6be1860c96524ad8b1436a2a.svg) 将相似度转化为概率，即 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 与点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) 中每个点的带权重的map关系。论文中称之为 _soft pointer、soft matching_。

**4.4 SVD Module**

得到点与点的对应关系后，下一步就是计算出到底该如何调整 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94.svg) ，使其尽量与 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) 对齐。

对于点 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-55abd6ee2c07065bfd8875b4e79a1cf9.svg) ，使用上面计算出来的与点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) 中每个点匹配的概率，加权求和，计算出来一个平均点：

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-499842f8e348e5dcb121fbd3112691a1.svg)

其中， ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-a58eef5e90227bfc61bdd11835af4d81.svg) 表示点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) 中N个点的坐标。 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-ff3b399485568c1abcd79f34d4ebcf3f.svg) 就是 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-55abd6ee2c07065bfd8875b4e79a1cf9.svg) 调整的目标点。按相同的方法，可以计算出点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94.svg) 中每个点的目标点。最后使用SVD得出 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-81651914e72da52a52f3baf300379792.svg) 的旋转、平移矩阵： ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-04e729b6b46ae88ef4a7c80ada1f5615.svg) 。

**4.5 Loss**

整个网络相当于输入点云 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-bb713738c1c0c8c066224f6258d3ff94.svg) 、 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-5080b1a67cc21b4d7d0e758935e59275.svg) ，输出 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-04e729b6b46ae88ef4a7c80ada1f5615.svg) 。用 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-04e729b6b46ae88ef4a7c80ada1f5615.svg) 与ground truth值来构建Loss：

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-54d733a26a690e190d1f020006f710d7.svg)

Loss中的第一项：因为希望 ![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-e645338ac2b74527c2f0e18e15138f50.svg) 是单位正交矩阵（Orthogonal Matrix），所以它的转置和逆应该相同。因此，理想情况下，以下等式应该成立：

![[公式]](%E6%B7%B1%E5%BA%A6paper/1625904000-654ceab18a1a115b4d3561c77be25329.svg)

## 实验内容以及分析

两个版本V1、V2：无注意力机制的为V1

在未见点云上进行测试：

![image-20200714180617238](%E6%B7%B1%E5%BA%A6paper/1625904345-e6288dcdf3d4c9c027c45967862856df.png)

在未见类别的点云上进行测试：

![在这里插入图片描述](%E6%B7%B1%E5%BA%A6paper/1625904345-b55b5ef73bff45267f8d10a20cfdb4d5.png)

鲁棒性测试：使用有噪声的输入

![在这里插入图片描述](%E6%B7%B1%E5%BA%A6paper/1625904345-cfa2b0e6c62cb0eeabaf212626f54ad0.png)

速度测试：ICP很快，可以看出其复杂度不高于线性，DCP在4096之前都很快，但是到4096突然变慢了10倍，复杂度应该高于线性。

![image-20200714180749728](%E6%B7%B1%E5%BA%A6paper/1625904345-62d8b5e9ce5cb5936271c767242876ad.png)

PointNet 与 DGCNN的对比：选择DGRCNN没错

![在这里插入图片描述](%E6%B7%B1%E5%BA%A6paper/1625904345-a7560c6accebe93f089995939712f7c3.png)

MLP与SVD分解的对比： MLP是一个通用逼近器，因此与ＳＶＤ做了一下对比：选择SVD没错

![image-20200714181206930](%E6%B7%B1%E5%BA%A6paper/1625904345-507d43b8bba756c8c26bebaf84e7f978.png)

PointNet作者指出特征维度对模型的精度是有很大影响的，越大越好，但是

高于阈值之后就影响甚微了。实验结果显示1024维优于512维度。这里作者并没有找到最优阈值，个人觉得用grid search找到最佳阈值然后再测试精度可能更好？

![image-20200714181358783](%E6%B7%B1%E5%BA%A6paper/1625904345-4a07ba2fad7f326ddbd9b9a7f4dc5b29.png)

最后作者提使用ICP作为后优化策略，即 使用DCP+ICP 可以得到更好的结果，但是只给出了示意图没有误差结果。

![在这里插入图片描述](%E6%B7%B1%E5%BA%A6paper/1625904345-ed69de935920083ba1b9f1d1db7d7f7f.png)

## 总结

论文将Transformer应用到了点云Registration问题中，通过Transformer中的attention机制，计算出一个“假想的目标点云“，这个**假想的目标点云**与**待调整点云**之间点的对应关系已知（soft matching）。再通过Loss约束，间接使得**假想的目标点云**向**真正的目标点云**不断逼近，最终实现点云的对齐匹配。

