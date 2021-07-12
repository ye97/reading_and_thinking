---
typora-root-url: pytorch-learn
---




title: pytorch_learn
categories:
  - python学习
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-05-22 20:19:12
urlname:
tags:
typora-root-url: pytorch-learn



# 线性回归

```python
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

# matplotlib绘图

~~~python 
#numpy.meshgrid()——生成网格点坐标矩阵。
~~~

![点集](/20180809112934345.png)

A,B,C,D,E,F是6个网格点，坐标如图，**坐标矩阵**得到为：

x=[[0,1,2],	y=[[0,0,0],

​	 [0,1,2]] 		 [1,1,1]]

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



![meshgrid](/meshgrid.png)

# Gradient Descent  

> 网络架构：
>
> 1，forward，直接从输入x写到返回y，
>
> 2，cost函数；
>
> 3，optimistic优化函数//此处为梯度下降函数
>
> 4，流程
>
> ​		forward
>
> ​		cost
>
> ​		optimistic

```python

x_data = [1.0, 2.0, 3.0] 
y_data = [2.0, 4.0, 6.0]
w = 1.0
def forward(x): 
    return x * w

def cost(xs, ys): 
    cost = 0
    for x, y in zip(xs, ys): 
        y_pred = forward(x)
        cost += (y_pred - y) ** 2 
     return cost / len(xs)

def gradient(xs, ys): 
    grad = 0
	for x, y in zip(xs, ys):
		grad += 2 * x * (x * w - y)
    return grad / len(xs)

print('Predict (before training)', 4, forward(4)) 
for epoch in range(100):
	cost_val = cost(x_data, y_data) 
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
	print('Epoch:', epoch, 'w=', w, 'loss=', cost_val) 
    print('Predict (after training)', 4, forward(4))

```

# pytorch基础 

## 基本单位-Tensor

### tensor和Tensor区别

​    Ternsor可以理解为一种数据结构，这种数据结构中包括data和grad是数据和梯度。此处注意一个小问题，**Tensor和tensor的区别**。torch.Tensor()是python类，初始化tensor类变量的。tensor是一个函数，可以将其他类型变量转换为Tensor类变量。

```python
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
```

## 使用pytorch实现线性回归y=wx

```python
import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = torch.Tensor([1.0]) # w的初值为1.0
w.requires_grad = True # 需要计算梯度
#传播图过程需要计算w的梯度 
#书写函数应该直接使用变量名和传播图即可
def forward(x):
    return x*w  # w是一个Tenso，Tensor参加的运算得到也是Tensor
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
        w.data = w.data - 0.01 * w.grad.data #权重更新时，需要用到标量，注意grad也是一个tensor
        w.grad.data.zero_() # after update, remember set the grad to zero
 
   print('progress:',epoch,l.item())# 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
 
print("predict (after training)", 4, forward(4).item())
```

## wx+b

```python
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

#pytorch实现wx+b的求导
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
#需要w求导
w.requires_grad=True

b=torch.Tensor([1])
b.requires_grad=True
#准备数据
for i in range(10):
    X.append(random()*10+1)
    Y.append(X[i] * 2 + 0.5)

for epoch in range(10000):
    for x, y in zip(X, Y):
        l=loss(w,b,x,y)
        #pytorch反向传播自动求解
        l.backward()
        #求data引用
        w.data=w.data-0.001*w.grad.data
        b.data=b.data-0.001*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

print(w.item())
print(b.item())
```

## 复杂结构设计

> prepare dataset//明确数据的维度和含义；样本矩阵应该写成   个数*特征
>
> Design model using Class //inherit from nn.Module  
>
> construct loss and optimizer  //构建优化器，也就是损失函数，都是使用api
>
> train cycle//循环训练，实际调用
>
> ​	forward//正向传播，l=loss(w,b,x,y)
>
> ​	backward//反向传播，l.backward()
>
> ​	update//更新参数。b.grad.data.zero_()

复杂线性设计：

### 构造model

> 继承module 
>
> 必须实现 init函数，调用父类的初始化函数 __init__ 函数（super语句），返回自己的构造类 torch.nn.Linear(1, 1)，这个构造类都是callable，linear是call函数 ，可以直接被调用
>
> 正向传播
>
> 反向传播
>
> 更新参数

```python
class LinearModel(torch.nn.Module):
    #nn表示neturnal network 
    #module 表示网络的包
    def __init__(self):
        #调用父辈的类函数的init函数  super（类，类实例）.父类方法（）
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        #torch.nn.Linear(1, 1) 构造一个对象返回给实例
        #Linear（Size of each input sample，Size of each output sample）
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        #实现foreward函数
        #此处linear就是一个callalbe对象，实现了__call__函数
        #self.linear没有括号就是一个可调用对象
        y_pred = self.linear(x)
        return y_pred
```

![](/Snipaste_2021-05-22_21-52-10.png)

> linear函数参数理解；X输入为N*Feature；O输出N**feature；

### 构造损失函数和优化函数

```python
# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
#损失函数还是在网络nn模块中
criterion = torch.nn.MSELoss(reduction='sum')
#sgd梯度下降，在optim优化器中，（模型参数求取就是在优化器中）
#model.parameters（）自动识别所有参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作
```

### 完整代码实现线性网络

```python
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
#传入model的参数他知道推导计算图
optim=torch.optim.SGD(model.parameters(),lr=0.01)
#参数的输出还是使用model
#model对象中有linear对象，linear对象中有w和b


#准备数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

for epoch in range(1,1000):
    #model是一个实例，但是他是__call__，可以调用，像函数一样可以调用的实例，pytorch中将类实例的call函数实例化了forward
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

```

## **Pytorch中tensor维度和torch.max()函数中dim参数的理解**

* * *

### 维度

参考了 https://blog.csdn.net/qq_41375609/article/details/106078474 ，  
对于torch中定义的张量，感觉上跟矩阵类似，不过常见的矩阵是二维的。当定义一个多维的张量时，比如使用 a =torch.randn(2, 3, 4) 创建一个三维的张量，返回的是一个

```python
[
     [
            [-0.5166,  0.8298,  2.4580, -1.9504],

            [ 0.1119, -0.3321, -1.3478, -1.9198],

            [ 0.0522, -0.6053,  0.8119, -1.3469]
     ],
     
     [
            [-0.3774,  0.9283,  0.7996, -0.3882],

            [-1.1077,  1.0664,  0.1263, -1.0631],

            [-0.9061,  1.0081, -1.2769,  0.1035]
   ]
]
```

当使用 a.size() 返回维度结果时，结果为 torch.Size(\[2, 3, 4\])，这里面有三个数值，数值的个数代表维度的个数 ，所以这里有三个维度，可以理解为一个中括号代表一个维度。数值 2 处在第一个位置，第一个位置代表是第一维度,2代表这个维度有两个元素，也就是第一个 \[ \] 里面两个元素，3代表在第二个维度，也就是在第一个 \[ \] 中的两个元素里面，又有三个元素，依次类推。这里格式十分固定，一旦定义，必须是一个元素里面有两个元素，这两个元素中每个再包含三个元素，再包含，依此类推，否则会报错。类似与树，维数等于相似的树的深度-1（以根为第一层），每一层就是一维。  
如生成一个

```python
torch.tensor(
    [
        [
            [1, 2, 3, 4]

            [3, 4, 2, 1]

            [4, 1, 2, 3]
        ]

        [
            [2, 1, 3, 4]

            [3, 4, 2, 1]

            [4, 1, 2, 3]
        ]
    ]
)
```

方便理解，以下图的形式展示，这里竖线代表一个维度，竖线上所有节点代表同一维度的所有元素。在下面所有图中，同颜色的元素都是按照从上往下按顺序排列的。  
![在这里插入图片描述](/1626078158-4076ace292a6253fc32dce45df07ee81.png)

* * *

### dim参数

在使用torch.max()函数和其他的一些函数时，会有dim这个参数。官网中定义使用torch.max()函数时，生成的张量维度会比原来的维度减少一维，除非原来的张量只有一维了. 要减少消去的是哪一维便是由dim参数决定的，dim参数实际上指的是我们计算过程中所要消去的维度。因为在比较时必须要指定使用哪些数字来比较 ，或者进行其他计算，比如 max 使一些数据中只要大的，sum只取和的结果，自然就会删减其他的一些数据从而引起降维。

* * *

以上面生成的三维的张量为例子，有三个维度，但是维度的数字顺序是 dim = 0， 1， 2；  
当指定torch.max(a，dim=0)时，也就是要删除第一个维度，删除第一个维度的话，那还剩下两个维度，也就是dim =1 ，2 。 剩下的两个维度的参数是 3 和 4，那么删除第一个维度后应该剩下torch.tensor(3, 4)这样形式的张量, dim参数可以使用负数，也就是负的索引，与列表中的索引相似，在本例中dim = -1 与dim = 2是一样的。  
返回的

```python
values=tensor([[-0.3774,  0.9283,  2.4580, -0.3882],
               [ 0.1119,  1.0664,  0.1263, -1.0631],
               [ 0.0522,  1.0081,  0.8119,  0.1035]]),
indices=tensor([[1, 1, 0, 1], 
                [0, 1, 1, 1],
                [0, 1, 0, 1]]))
```

从返回的结果看是这种形式，产生这种结果是因为删除了第一个维度那么该返回 3 \* 4 这种二维的张量，第一维中两个元素的形式正好是 3 \* 4， 那么就将这个维度的两个子元素中的相应的位置的值比较一下大小，那么会生成一个新的 3 \* 4 的张量，再返回一下正好可以，indices记录的是 "在比较中胜利的元素“ 原来所属的元素的位置。例如在第一个位置上，-0.3774比 -0.5166大，所以返回-0.3774，-0.3774是在第一维度里面的第二个元素的位置上，这个位置索引为1.剩下的位置的同理。

用树状图理解  
![在这里插入图片描述](/1626078158-ebfb3274f6d6715e9b29c2f6c1e662fa.png)

图中的不同颜色的三个子元素，在相同位置比较，大的返回形成新的元素，其他位置同理。那么黑色的维度 dim = 1 也就消除了.

* * *

dim = 0时，如图，两个3\*4的子元素张量 相对应的位置 比较大小，剩下一个3 \* 4的二维张量  
![在这里插入图片描述](/1626078158-937110c3fb27f857540925eef3c2e71c.png)

当dim = 2或者 dim = -1，删除的是最后一个维度，在这个例子中吗，将所有的第三维的子元素最大的值返回，返回2 \* 3，看起来就像是找所在矩阵一行里面的最大值一样。

```python
values=tensor([[2.4580, 0.1119, 0.8119],
        [0.9283, 1.0664, 1.0081]]),
indices=tensor([[2, 0, 2],
        [1, 1, 1]]))
```

* * *

举一个sum()例子,当使用上述使用torch.sum（a，dim = 1），消去第二个维度，剩下一，三维度，也就是2 \* 4形状的张量。将第二维上面的三个子元素相同位置的相加，第二维也就不见了，第一维中的两个元素的子元素就从3\*4形成了一个1 \*4的，总的形状就变成了2 \* 4

```python
tensor([[-0.3525, -0.1076,  1.9221, -5.2171],
        [-2.3912,  3.0028, -0.3510, -1.3478]])
```

* * *

再举一个例子，使用torch.randn(2, 3, 4, 5) 创建一个四维张量，使用torch.max(dim=-3)，也就是torch.max(dim=1)

```python
torch.tensor([[[[ 0.7106,  1.3332, -1.0423, -0.1609, -0.2846],
	          [ 0.6400,  2.2507, -0.5740, -0.9986,  0.0066],
	          [-0.0527,  1.4097, -0.4439,  0.4846,  1.5418],
	          [ 1.0027,  0.9398,  1.5202, -1.1660, -0.1230]],
	
	         [[ 0.5725, -1.7838, -0.7320, -1.4419,  1.5762],
	          [ 0.6407,  0.0527,  1.7005,  1.6350, -0.2610],
	          [ 1.3307, -0.3210, -1.7203,  0.9050,  0.2442],
	          [ 0.9418, -0.1511,  0.8248, -0.0786, -0.6153]],
	
	         [[ 1.0182,  0.3190, -0.3408, -2.1801, -0.3931],
	          [ 1.2325, -0.3304,  1.0116,  0.0791, -1.1174],
	          [ 0.2331, -0.9062,  0.5680,  1.6061, -1.0933],
	          [ 0.6935, -0.5140, -0.5178,  1.2557,  0.2319]]],
	
	
	        [[[ 1.0916,  0.7171, -0.7936,  1.1741, -0.5457],
	          [-0.6541, -0.6720, -0.7892, -0.6961, -1.1030],
	          [ 1.8680, -0.1746,  0.8455, -1.1021,  0.6855],
	          [ 1.2070, -0.6152, -1.3345, -0.0724,  1.2062]],
	
	         [[-0.5130, -0.5510, -0.8278, -0.2279, -1.4425],
	          [ 0.2073,  1.3065, -0.0326, -1.2566,  0.6097],
	          [-1.0413,  1.2638, -0.8479, -0.0353, -0.7191],
	          [ 0.0662,  0.7683,  0.2145, -0.0988, -2.3348]],
	
	         [[ 0.6631, -0.0040, -0.0681,  1.1681,  1.3904],
	          [-0.1761,  1.4668,  0.9670, -0.5629,  0.2941],
	          [-0.6235,  0.1844, -0.4321, -0.0581, -0.9352],
	          [ 0.1717, -0.9188,  0.3014, -0.0734, -0.1324]]]])
	         
```

在这里面，当dim = 1，也就是要动第二个维度手，那么删掉它后剩下torch.randn(2，4， 5)形式，那么就  
\[\[ 0.7106, 1.3332, -1.0423, -0.1609, -0.2846\],  
\[ 0.6400, 2.2507, -0.5740, -0.9986, 0.0066\],  
\[-0.0527, 1.4097, -0.4439, 0.4846, 1.5418\],  
\[ 1.0027, 0.9398, 1.5202, -1.1660, -0.1230\]\]

和  
\[\[ 0.5725, -1.7838, -0.7320, -1.4419, 1.5762\],  
\[ 0.6407, 0.0527, 1.7005, 1.6350, -0.2610\],  
\[ 1.3307, -0.3210, -1.7203, 0.9050, 0.2442\],  
\[ 0.9418, -0.1511, 0.8248, -0.0786, -0.6153\]\]  
还有  
\[\[ 1.0182, 0.3190, -0.3408, -2.1801, -0.3931\],  
\[ 1.2325, -0.3304, 1.0116, 0.0791, -1.1174\],  
\[ 0.2331, -0.9062, 0.5680, 1.6061, -1.0933\],  
\[ 0.6935, -0.5140, -0.5178, 1.2557, 0.2319\]\]

这三个子元素相应为位置比较大小，大的留下，生成新的张量，列如对于第一个位置，1.0182 比 0.5725 和 0.7106 大，所以它留下，它在元素在要是动手的维度里面的位置索引为2，其它同理  
但是这个维度还之前还有一个维度，那么只要对所有的同维度的做相同操作就可以了，所以返回之如下

```python
values=tensor([[[ 1.0182,  1.3332, -0.3408, -0.1609,  1.5762],
         [ 1.2325,  2.2507,  1.7005,  1.6350,  0.0066],
         [ 1.3307,  1.4097,  0.5680,  1.6061,  1.5418],
         [ 1.0027,  0.9398,  1.5202,  1.2557,  0.2319]],

        [[ 1.0916,  0.7171, -0.0681,  1.1741,  1.3904],
         [ 0.2073,  1.4668,  0.9670, -0.5629,  0.6097],
         [ 1.8680,  1.2638,  0.8455, -0.0353,  0.6855],
         [ 1.2070,  0.7683,  0.3014, -0.0724,  1.2062]]]),
indices=tensor([[[2, 0, 2, 0, 1],
         [2, 0, 1, 1, 0],
         [1, 0, 2, 2, 0],
         [0, 0, 0, 2, 2]],

        [[0, 0, 2, 0, 2],
         [1, 2, 2, 2, 1],
         [0, 1, 0, 1, 0],
         [0, 1, 2, 0, 0]]]))
```

# 逻辑回归

```python
#准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
#                                                                                          #
#准备类使用api
class LogisticRegressionModel(torch.nn.Module):
	def    init  (self):
        super(LogisticRegressionModel, self).  init  ()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
       y_pred = F.sigmoid(self.linear(x))
       return y_pred
#类实例
model = LogisticRegressionModel()

#构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#train cycle
for epoch in range(1000):
	y_pred = model(x_data)
	loss = criterion(y_pred, y_data)
	print(epoch, loss.item())
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

```





# 数据集处理

> 1，概念 epoch：一次所有数据集被处理，包括前向和反向
>
> 2，batch_size；一次被处理的样本数量
>
> 3，iteration ：迭代次数，总样本除以batch_size
>
> 主要就是dataset定义和data-loader

## Dataset

> ​	继承from torch.utils.data import Dataset
>
> ​    实现初始化，获取条目，返回长度

```python
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class DiabetesDataset(Dataset):
    def __init__(self):
    	pass
    def __getitem__(self, index):
    	pass
    def __len__(self):
    	pass
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
						  batch_size=32,
						  shuffle=True,
						  num_workers=2)
```

datasetloader主要作用如下图：

![](/Snipaste_2021-05-22_22-02-28.png)

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
batch_size=32,
shuffle=True,
num_workers=2)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
model = Model()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1. Prepare data
        inputs, labels = data
        # 2. Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3. Backward
        optimizer.zero_grad()
        loss.backward()
        # 4. Update
        optimizer.step()
```

# 多分类

```python
#mnist多分类任务one-hot,以Linear全连接的方式，进行分类
import torch
from torchvision import transforms#打包函数的变化
from torchvision import datasets#
from torch.utils.data import DataLoader
import torch.nn.functional as F#函数模块
import torch.optim as optim
batch_size = 64

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))#你均值和方差都只传入一个参数，就报错了.
    # 这个函数的功能是把输入图片数据转化为给定均值和方差的高斯分布，使模型更容易收敛。图片数据是r,g,b格式，对应r,g,b三个通道数据都要转换。
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                                train=True,
                                download=True,
                                transform=transform)
train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                                train=False,
                                download=True,
                                transform=transform)
test_loader = DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 =torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx , data in enumerate(train_loader, 0):
        inputs,target=data
        optimizer.zero_grad()
        outputs = model(inputs)#outputs:64*10,行表示对于图片的预测，batch=64
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx %300 ==299:
            print('[%d,%5d] loss: %.3f'%(epoch+1,batch_idx+1,running_loss/300))
            running_loss=0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data,dim=1)
            total+=labels.size(0)#每一批=64个，所以total迭代一次加64
            correct +=(predicted==labels).sum().item()
    print('Accuracy on test set:%d %%'%(100*correct/total))


if __name__ =="__main__":
    for epoch in range(10):
        train(epoch)#封装起来，若要修改主干就很方便
        test()
```



# 各种函数

## softmax

![](/Snipaste_2021-05-22_22-13-22.png)

![Snipaste_2021-05-22_22-13-36](/Snipaste_2021-05-22_22-13-36.png)

交叉熵=softmax+onthot

##  numpy.random.uniform(low,high,size)，

随机采样功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.

参数介绍:
low: 采样下界，float类型，默认值为0；
high: 采样上界，float类型，默认值为1；
size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m\*n\*k个样本，缺省时输出1个值。

## 乘法函数

元素乘法：np.multiply(a,b)
矩阵乘法：np.dot(a,b) 或 np.matmul(a,b) 或 a.dot(b) 或直接用 a @ b !
唯独注意：*，在 np.array 中重载为元素乘法，在 np.matrix 中重载为矩阵乘法!

## 截取函数

```
np.clip(
	a, 
	a_min, 
	a_max, 
	out=None):
a：输入矩阵；
a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；
a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；
out：可以指定输出矩阵的对象，shape与a相同
```

## 随机函数（待完成）

[(1条消息) numpy.random.randn()用法_u012149181的博客-CSDN博客_np.random.randn](https://blog.csdn.net/u012149181/article/details/78913167)

## apply函数



```python
DataFrame.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
1.该函数最有用的是第一个参数，这个参数是函数，相当于C/C++的函数指针。
2.这个函数需要自己实现，函数的传入参数根据axis来定，比如axis = 1，就会把一行数据作为Series的数据
结构传入给自己实现的函数中，我们在函数中实现对Series不同属性之间的计算，返回一个结果，则apply函数
会自动遍历每一行DataFrame的数据，最后将所有结果组合成一个Series数据结构
并返回。
3.apply函数常与groupby函数一起使用，如下图所示


data=np.arange(0,16).reshape(4,4)
data=pd.DataFrame(data,columns=['0','1','2','3'])
def f(x):
    return x.max()
print(data)
print(data.apply(f,axis=1))
    0   1   2   3
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
0     3
1     7
2    11
3    15
dtype: int64

```

![在这里插入图片描述](/apply)

# 各种报错

## 没有cuda即没有gpu设备

```python
#一开始的位置添加下面语句
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#后面的修改
cuda()的地方改成.to(device)
```

# 概念问题

## encoder和decoder

encoder-decoder模型，又叫做编码-解码模型。这是一种应用于seq2seq问题的模型**。seq2seq表示序列对序列。

所谓编码，就是将输入序列转化成一个固定长度的向量；解码，就是将之前生成的固定向量再转化成输出序列。 

两个问题：一是语义向量无法完全表示整个序列的信息，还有就是先输入的内容携带的信息会被后输入的信息稀释掉，或者说，被覆盖了。输入序列越长，这个现象就越严重。

# 各种网络结构

## Res2Net模块介绍

> 如图所示，Res2Net网络中的主要结构的思想是将原来残差卷积中的3×3卷积层接收到的来自输入层1×1卷积后的特征图分解为四部分，第一部分不进行操作，第二部分通过一个3×3卷积层，第三部分在通过一个3×3卷积层前与第二部分卷积后的特征图相加，第四部分在通过一个3×3卷积层前与第三部分卷积后的特征图相加，最终将得到的四个部分的特征图进行拼接成与输入层输出同样层数的特征图再送到输出层做1×1卷积，具体如下图所示。

![ResNet和Res2Net](/1625901456-1f50ac740b8c20ef54ad04392bf2531c.jpg)

#### Pytorch代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as function


class Res2NetBlock(nn.Module):
    def __init__(self, inplanes, outplanes, scales=4):
        super(Res2NetBlock, self).__init__()

        if outplanes % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')

        self.scales = scales
        # 1*1的卷积层
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, 32, 1, 1, 0),
            nn.BatchNorm2d(32)
        )
        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8)
        )
        # 1*1的卷积层
        self.outconv = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input = x
        x = self.inconv(x)

        # scales个部分
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(xs[0])
        ys.append(function.relu(self.conv1(xs[1])))
        ys.append(function.relu(self.conv2(xs[2]) + ys[1]))
        ys.append(function.relu(self.conv2(xs[3]) + ys[2]))
        y = torch.cat(ys, 1)

        y = self.outconv(y)
        
        output = function.relu(y + input)

        return output
```

## cnn设计

[cnn理解](D:\git_rep\hexo\source\_posts\cnn理解.md)