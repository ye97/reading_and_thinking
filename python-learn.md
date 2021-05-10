---
title: python_learn
categories:
  - python
tags:
  - 笔记
  - 总结
  - 语法
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-05-10 09:26:07
urlname:

---



# 继承

~~~python
 class A(B):
 #A继承B
#重写父类方法
def 函数（self，参数）：
#self是实例，定义时需要写，调用自动传入
def __init__(self):
     #调用父辈的同名方法都是这样的调用 super（本类名，self）.方法
    super(self,Model).__init__()
#python中的super( test, self).init()
#首先找到test的父类（比如是类A），然后把类test的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数.
~~~

# 魔法函数

​	**魔法函数是在类中以双下划线开头，双下划线结尾的，python本身给我们定义了许多这种方法，让我们能在类中直接使用。**

​	魔法函数就是函数操作的重定义，比如getitem，当python执行键对操作时就执行这个函数

如果类把某个属性定义为序列，可以使用__getitem__()输出序列属性中的某个元素.如下将employee设置为list。

~~~python
#遍历实例中所有元素
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list

company = Company(['tom', 'bob', 'jane'])
employee = company.employee

for em in employee :
    print(em)

~~~

~~~python
class Company(object):
    def __init__(self, employee_list):
        self.employee = employee_list
	#item是序列号
    def __getitem__(self, item):
        return self.employee[item]

company = Company(['tom', 'bob', 'jane'])

for em in company:
    print(em)

~~~

# 内置函数

~~~python
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
~~~

# numpy库

## cumsum

~~~python
arr = np.array([[[1,2,3],[8,9,12]],[[1,2,4],[2,4,5]]]) #2*2*3
print(arr)

print(arr.cumsum(0))
print(arr.cumsum(1))
print(arr.cumsum(2))
#arr输出
[[[ 1  2  3]
  [ 8  9 12]]

 [[ 1  2  4]
  [ 2  4  5]]]
#对cumsum（0）表示0维度操作，arr[0]加到arr[1]
#arr.cumsum(0)，上面块加到下面
[[[ 1  2  3]
  [ 8  9 12]]

 [[ 2  4  7]
  [10 13 17]]]
#cumsum(1)
[[[ 1  2  3]
  [ 9 11 15]]
 
 [[ 1  2  4]
  [ 3  6  9]]]
#cumsum(2)
[[[ 1  3  6]
  [ 8 17 29]]
 
 [[ 1  3  7]
  [ 2  6 11]]]
#0去第一个轴，按轴累加
#其他同理
~~~

## np.concatenate

~~~python
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[11,21,31],[7,8,9]])
print(np.concatenate((a,b),axis=1))
#[[ 1  2  3 11 21 31]
#[ 4  5  6  7  8  9]]
~~~

## np.zeros

##  np.sort

## **numpy.arange([start, ]stop, [step, ]dtype=None)**

在给定的时间间隔内返回均匀间隔的值。

# argparse库

利用 argparse 存参数

~~~python
def parameter():
    parser = argparse.ArgumentParser(description='TextCNN model parameter ')
    parser.add_argument('-batch_size',default=10,help='the train batch size')
    parser.add_argument('-epoch',default=5,help='the time need train')
    parser.add_argument('-learing_rate',default=1e-3)
    parser.add_argument('-embed_size',default=100,help='the word dimensionality')
    parser.add_argument('-output_size',default=2)
    parser.add_argument('-k',default=10,help='the cross validation size')
    parser.add_argument('-dropout_rate',default=0.1,help='dropout layer')
    args = parser.parse_args() 
    
    return args

~~~

```python
from  untitled1 import parameter
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
 
args = parameter() #参数
 
x = torch.rand(100,25,args.embed_size) #100个句子,25个词,embed_size
label = [0] *50 +[1]*50
label = torch.tensor(label,dtype=torch.long)
 
 
class DataSet(Dataset):
    def __init__(self,X,label):
        self.x_data = X
        self.y_data = label
        self.len = len(label)
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len
    
class textCNN(nn.Module):
    def __init__(self,words_num):
        super(textCNN, self).__init__()
        self.words_num = words_num
        self.embed_size = args.embed_size  #args
        self.class_num = args.output_size  #args
        self.drop_rate = args.dropout_rate
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,3,(3,self.embed_size)),
            nn.BatchNorm2d(3)) ###in_channels, out_channels, kernel_size
        self.conv2 = nn.Sequential(
            nn.Conv2d(1,3,(4,self.embed_size)),
            nn.BatchNorm2d(3)) 
        self.conv3 = nn.Sequential(
            nn.Conv2d(1,3,(5,self.embed_size)),
            nn.BatchNorm2d(3))
        
        self.max_pool1 = nn.MaxPool1d(5)
        self.max_pool2 = nn.MaxPool1d(4)
        self.max_pool3 = nn.MaxPool1d(3)
        
        self.dropout = nn.Dropout(self.drop_rate)
        self.linear = nn.Linear(48,self.class_num) ##后面算出来的
        # 3 -> out_channels 3 ->kernel_size 1 ->max_pool
    
    def forward(self,sen_embed): #(batch,max_len,embed_size)
        sen_embed = sen_embed.unsqueeze(1) #(batch,in_channels,max_len,embed_size)
        
        conv1 = F.relu(self.conv1(sen_embed))  # ->(batch_size,out_channels,output.size,1)
        conv2 = F.relu(self.conv2(sen_embed))
        conv3 = F.relu(self.conv3(sen_embed))
        
        conv1 = torch.squeeze(conv1,dim=3)
        conv2 = torch.squeeze(conv2,dim=3)
        conv3 = torch.squeeze(conv3,dim=3)
        
        x1 = self.max_pool1(conv1)
        x2 = self.max_pool2(conv2)
        x3 = self.max_pool3(conv3) ##batch_size,out_channel,18
        
        x1 = x1.view(x1.size()[0],-1) ###batch_size 不能等于1
        x2 = x2.view(x2.size()[0],-1)
        x3 = x3.view(x3.size()[0],-1)
        
        x = torch.cat((x1,x2),dim=1)
        x = torch.cat((x,x3),dim=1)
        output = self.linear(self.dropout(x))
        
        return output 
 
dataset =  DataSet(x, label) 
data_loader = DataLoader(dataset,args.batch_size, shuffle=True) 
 
model = textCNN(25) #word_num
 
loss_function = nn.CrossEntropyLoss()
def train(args,model,data_loader,loss_function):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learing_rate)
    criterion = loss_function
    model.train() 
    for epoch in range(args.epoch): ##2个epoch
        for step,(x,target) in enumerate(data_loader): 
            output = model(x)
            loss = criterion(output,target)
            optimizer.zero_grad()
            #loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()
        print(epoch)
train(args,model,data_loader,loss_function)

```

