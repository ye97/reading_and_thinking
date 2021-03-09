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

## 5.2 pytorch实现简单y=wx求取权重，第二部分为wx+b

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

## 5.3 pytorch实现复杂网络基本结构

#### 		1.  prepare dataset

##### 						1,明确样本维度和标签维度

##### 						2，要求取量的维度设置（w和b为例）

##### 						3，样本矩阵应该写成 个数*特征

##### 						4，要求导的参数应该反推得到

#### 		2. design model （inhert from module）

##### 						1，唯一目标就是求到输出值

#### 		3.construct loss and opitimize

##### 						1，计算loss是为了进行反向传播，

##### 						2，optimizer是为了更新梯度

#### 		4.train cycle

##### 						1.forwad

##### 						2.backward

##### 						3.update

## 5.4调用pytorch实现线性回归

### 		5.4.1 model构造

#### 		1，继承module 

#### 		2，必须实现init函数，调用父类的初始化函数 __init__ 函数（super语句），返回自己的构造类 torch.nn.Linear(1, 1)，这个构造类都是callable，linear是call函数 ，可以直接被调用

#### 		3，必须实现自己的forward计算图

~~~python 
class LinearModel(torch.nn.Module):
    #nn表示neturnal network 
    #module 表示网络的包
    def __init__(self):
        #调用父辈的类函数的init函数  super（类，类实例）.父类方法（）
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        #torch.nn.Linear(1, 1) 构造一个对象返回给实例
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        #实现foreward函数
        #此处linear就是一个callalbe对象，实现了__call__函数
        y_pred = self.linear(x)
        return y_pred
~~~

### 5.4.2 构造损失函数和优化函数

~~~python
# construct loss and optimizer

# criterion = torch.nn.MSELoss(size_average = False)

#损失函数还是在网络nn模块中
criterion = torch.nn.MSELoss(reduction='sum')
#sgd梯度下降，在optim优化器中，（模型参数求取就是在优化器中）
#model.parameters（）自动识别所有参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作
~~~

### 5.4.3模型求取

#### 1，模型调用求取得到y_pre

#### 2,   求取loss值

#### 3.1 反向传播前清零梯度

#### 3， 反向传播

#### 3.1.2反向传播后更新参数

~~~ python
#完整代码
import torch
#model实现自己的类，继承module模块
class Model(torch.nn.Module):
    #继承他的原因是初始化的调用函数要使用他
    def __init__(self):
        #调用父辈的同名方法都是这样的调用 super（本类名，self）.方法
        # python中的super( test, self).init()
        # 首先找到test的父类（比如是类A），然后把类test的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数.
        super(Model,self).__init__()
        #实例中添加linear函数，使用torch中linear函数，返回得到一个linear对象
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        #实现正向传播图 调用实例对象的linear对象，即上面初始化的对象
        return self.linear(x)


model=Model()

#此处loss和optim都是可调用对象，即无参构造，传入参数调用
loss=torch.nn.MSELoss(reduction="sum")

optim=torch.optim.SGD(model.parameters(),lr=0.01)

#准备数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

for epoch in range(1,1000):
    #model是一个实例，但是他是__call__，可以调用，像函数一样可以调用的实例
    y=model(x_data)
    #loss同理，调用就是调用类中的函数
    cost=loss(y,y_data)
    print(epoch,":",cost.data.item(),cost.data.item())
    #反向传播前先清理梯度
    optim.zero_grad()
    #反向传播
    cost.backward()
    #更新参数（更新参数和清理梯度，都是优化器的工作）
    optim.step()
print(model.linear.weight.item())
print(model.linear.bias.item())
~~~

# 6，数据集处理

1，概念 epoch：一次所有数据集被处理，包括前向和反向

2，batch_size；一次被处理的样本数量

3，iteration ：迭代次数，总样本除以batch_size

## 6.1 dataset类重写

​			dataset类是pytorch里面的抽象类提供给我们构造自己的数据集



​		**目标：实现数据的结构（正则化之类的），索引（len和getitem）**

​		transform=transforms.compose(compose就是将所需的transforms的各种变换有序地组合在一起。)

```
transform = transforms.Compose
(	[transforms.ToTensor(), 
	transforms.Normalize((0.1307,), (0.3081,))
	])  # 归一化,均值和方差
```

### 			1.1____init____实现

### 			1.2 getitem

### 			1.3 len函数



## 6.2 dataloader类，pytorch已经提供了

​				DataLoader对数据集先打乱(shuffle)，然后划分成mini_batch。

### 			1.1 dataset名

### 			1.2 datasize

### 			1.3 shuffle

### 			1.4 num works（线程数）



## 6.3 维度变换

a.view()和.reshape() 丢失维度信息，直接变换维所输入的维度

squeeze(4)插入的维度就在4的位置，就是多添加一个维度



## 6.4 过拟合

添加正则化项

添加冲量（考虑历史梯度）

learing rate衰减

early stop 使用测试集的准确率，最高值

drop out 退出部分网络，nn.drop(0.5)退出百分之50













# 7，多分类

![sofrmax](D:\git_rep\hexo\source\_posts\pytorch_learn\sofrmax.png)

每个指标进行指数缩放然后除以全体和。

NLLLoss函数=log+onehot

![NLLLoss](D:\git_rep\hexo\source\_posts\pytorch_learn\NLLLoss.png)

crossemptoryloss函数

![CrossEntropyLoss](D:\git_rep\hexo\source\_posts\pytorch_learn\CrossEntropyLoss.png)

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
#transforms。compose（）就是一系列的操作组合
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差
#这个代码不能运行，但是dataset 操作可以参见官方帮助文档 #https://pytorch.org/docs/stable/nn.html#crossentropyloss

#root参数表示下载存放位置，train参数表示train训练集中提取，download表示下载
#dataset只是处理数据的结构
#dataloader：shuffle and  batchsize
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # -1其实就是自动获取mini_batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活，不进行非线性变换


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    #train_loader 按照batch_size划分批次，成为可迭代对象
    for batch_idx, data in enumerate(train_loader, 0):
        
        # 获得一个批次的数据和标签
        inputs, target = data
        optimizer.zero_grad()
        # 获得模型预测结果(64, 10)
        outputs = model(inputs)
        # 交叉熵代价函数outputs(64,10),target（64）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

# 8，CNN

## 1.1 卷积

![cnn](D:\git_rep\hexo\source\_posts\pytorch_learn\cnn.png)

## 1.2 卷积尺寸

![卷积维度](D:\git_rep\hexo\source\_posts\pytorch_learn\卷积维度.png)

输入 n表示channnel数，则输入是 （n，widthin，heightin）



卷积核：m个，每个都对输入卷积，每个尺寸为 （n，width_kernal，heightin_kernal）(每一个通道都要配一个核)

输出：(m,widthin-width_kernel+1,heightin-height_kernel+1) 

## 1.3 二维卷积构造

conv2d 2维卷积网络

class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

各个参数值：
stride：步长，卷积块的滑动步长

zero-padding:图像四周填0

groups:分组卷积

Convolution 层的参数中有一个group参数，其意思是将对应的输入通道与输出通道数进行分组, 默认值为1, 也就是说默认输出输入的所有通道各为一组。 比如输入数据大小为90x100x100x32，通道数32，要经过一个3x3x48的卷积，group默认是1，就是全连接的卷积层。

如果group是2，那么对应要将输入的32个通道分成2个16的通道，将输出的48个通道分成2个24的通道。对输出的2个24的通道，第一个24通道与输入的第一个16通道进行全卷积，第二个24通道与输入的第二个16通道进行全卷积。

极端情况下，输入输出通道数相同，比如为24，group大小也为24，那么每个输出卷积核，只与输入的对应的通道进行卷积。

bias:卷积后是否加偏移量

dilation:控制 kernel 点之间的空间距离

![conv2d_dilation](D:\git_rep\hexo\source\_posts\pytorch_learn\conv2d_dilation.png)

![conv2d_dilation2](D:\git_rep\hexo\source\_posts\pytorch_learn\conv2d_dilation2.png)

看下面灰色

