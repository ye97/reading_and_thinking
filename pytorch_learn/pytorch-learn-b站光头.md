---
title: pytorch_learn
categories:
  - 编程
  - 深度学习
tags:
  - pytorch
  - 深度
  - 总结
date: 2021-03-01 16:03:30

---



[toc]

# 1,scikit-learn 框架

# 2, 线性回归

``` python
#markdown添加代码 三个~键即可
#线性回归模型 
import numpy as np
import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
 
def forward(x):
    return x*w
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
 
# 穷举法
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
#plot 输入x和y的数组直接输出画图即可    
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()  
```

# 3,matplotlib绘图

~~~python 
#numpy.meshgrid()——生成网格点坐标矩阵。
~~~

![点集](D:\git_rep\hexo\source\_posts\pytorch_learn\20180809112934345.png)

A,B,C,D,E,F是6个网格点，坐标如图，**坐标矩阵**得到为：

x=[[0,1,2],	y=[[0,0,0],

​	[0,1,2]] 		 [1,1,1]]

{% asset_img 反向传播理解.png 反向传播 %}

{%asset_img 反向传播理解.png 图片描述%}

横坐标矩阵X中的每个元素，与纵坐标矩阵Y 中对应位置元素，共同构成一个点的完整坐标。如B点坐标( X12 , Y12) = (1,1) 

~~~python
import numpy as np
import matplotlib.pyplot as plt
#linspace 线段等分
x = np.linspace(0,1000,20)
y = np.linspace(0,500,20)
#网格等分
X,Y = np.meshgrid(x, y)

plt.plot(X, Y,
         color='limegreen',  # 设置颜色为limegreen
         marker='.',  # 设置点类型为圆点
         linestyle='')  # 设置线型为空，也即没有线连接点
#grid画网格，对应x，y连线
plt.grid(True)
plt.show()
~~~

![meshgrid](D:\git_rep\hexo\source\_posts\pytorch_learn\meshgrid.png)zh

# 4,反向传播理解

![反向传播理解](D:\git_rep\hexo\source\_posts\pytorch_learn\反向传播理解.png)

画出传播图来求导，注意loss函数作为一个整体还可以化简，添加一个r

# 5,pytorch基础

## 5.1基本单位-Tensor

### 5.1.1 tensor和Tensor区别

​    Ternsor可以理解为一种数据结构，这种数据结构中包括data和grad是数据和梯度。此处注意一个小问题，**Tensor和tensor的区别**。torch.Tensor()是python类，初始化tensor类变量的。tensor是一个函数，可以将其他类型变量转换为Tensor类变量。

5.1.2  tensor常见知识，type和grad属性

~~~python	
import torch
a = torch.Tensor([1.0])
a.requires_grad = True # 或者 a.requires_grad_()
print(a)
print(a.data)
print(a.type())             # a的类型是tensor
print(a.data.type())        # a.data的类型是tensor
print(a.grad)
print(type(a.grad))
#输出结果
/*
tensor([1.], requires_grad=True)
tensor([1.])
torch.FloatTensor
torch.FloatTensor
None
<class 'NoneType'>
*/

~~~

5.1.2 使用pytorch实现线性回归

~~~ python
import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = torch.Tensor([1.0]) # w的初值为1.0
w.requires_grad = True # 需要计算梯度
#传播图过程需要计算w的梯度 
#书写函数应该直接使用变量名和传播图即可
def forward(x):
    return x*w  # w是一个Tensor
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
#item()表示直接取出值，data表示对tensor中的值进行引用
print("predict (before training)", 4, forward(4).item())
 
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward() #  backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data   # 权重更新时，需要用到标量，注意grad也是一个tensor
 
        w.grad.data.zero_() # after update, remember set the grad to zero
 
    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
 
print("predict (after training)", 4, forward(4).item())
~~~

### 5.1.2 pytorch实现简单y=wx求取权重，第二部分为wx+b

~~~python
import torch
def forward(x,w):
    return w*x
#前向传播目标是损失函数
def loss(y,x):
    y_pre=forward(x,w)
    return (y-y_pre)**2

#前两个函数应该理解为计算图的构建，因此手动画个图然后来实现计算图，明确那几个参数是需要计算求取的

x_data=[1,2,3,4,5]
y_data=[2,1000,6,8,10]
#构造计算图中量都应该是tensor量
w=torch.Tensor([1])
w.requires_grad=True
#迭代1000词
for epoch in range(1,1000):
    #随机梯度下降，按顺序选取x和y
    for x,y in zip(x_data,y_data):
        l=loss(y,x)
        l.backward()
        #0.01表示学习率，如果学习率设置过容易发散
        #输出nan，表示not a number 就是表示无穷大不是数
        #inf表示无穷大
        #更新权重
        w.data=w.data-0.01*w.grad.data
        #消除权重的值
        w.grad.data.zero_()
print(w.item())


import  torch
def forward(w,b,x):
    return w*x+b
def loss(w,b,x,y):
    y_pre=forward(w,b,x)
    return (y-y_pre)**2

from random import random
X=[]
Y=[]
w=torch.Tensor([1])
w.requires_grad=True

b=torch.Tensor([1])
b.requires_grad=True
for i in range(10):
    X.append(random()*10+1)
    Y.append(X[i] * 2 + 0.5)

for epoch in range(10000):
    for x, y in zip(X, Y):
        l=loss(w,b,x,y)
        l.backward()
        w.data=w.data-0.001*w.grad.data
        b.data=b.data-0.001*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

print(w.item())
print(b.item())

~~~

### 5.1.3 pytorch实现复杂网络基本结构

#### 		1.  prepare dataset

##### 					1,明确样本维度和标签维度

##### 					2，要求取量的维度设置（w和b为例）

##### 					3，样本矩阵应该写成 个数*特征

##### 					4，要求导的参数应该反推得到

#### 		2. design model （inhert from module）

##### 					1，唯一目标就是求到输出值

#### 		3.construct loss and opitimize

##### 					1，计算loss是为了进行反向传播，

##### 					2，optimizer是为了更新梯度

#### 		4.train cycle

##### 					1.forwad

##### 					2.backward

##### 					3.update





