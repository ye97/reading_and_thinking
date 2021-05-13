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

------

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

# 文件操作

## 常见函数

```python
os.path.abspath(path) #返回绝对路径（包含文件名的全路径）

os.path.basename(path) —— 去掉目录路径获取(带后缀的)文件名

os.path.dirname(path) —— 去掉文件名获取目录

os.path.split(path) —— 将全路径分解为(文件夹,文件名)的元组

os.path.splitext(path)  #分割全路径，返回路径名和文件扩展名的元组

os.path.join()  #路径拼接

os.path.isdir()  #用于判断对象是否为一个全路径

os.path.isfile(path)  #判断文件是否存在？如果path是一个存在的文件，返回True。否则返回False。

os.path.isdir(path)  #判断目录是否存在？如果path是一个存在的目录，则返回True。否则返回False。

 

os.path.abspath(path) #返回绝对路径
 

os.path.basename(path) #返回文件名
 

os.path.commonprefix(list) #返回多个路径中，所有path共有的最长的路径。
 

os.path.dirname(path) #返回文件路径
 

os.path.exists(path)  #路径存在则返回True,路径损坏返回False
 

os.path.lexists  #路径存在则返回True,路径损坏也返回True
 

os.path.expanduser(path)  #把path中包含的"~"和"~user"转换成用户目录
 

os.path.expandvars(path)  #根据环境变量的值替换path中包含的”$name”和”${name}”
 

os.path.getatime(path)  #返回最后一次进入此path的时间。
 

os.path.getmtime(path)  #返回在此path下最后一次修改的时间。
 

os.path.getctime(path)  #返回path的大小
 

os.path.getsize(path)  #返回文件大小，如果文件不存在就返回错误
 

os.path.isabs(path)  #判断是否为绝对路径
 

os.path.isfile(path)  #判断路径是否为文件
 

os.path.isdir(path)  #判断路径是否为目录
 

os.path.islink(path)  #判断路径是否为链接
 

os.path.ismount(path)  #判断路径是否为挂载点（）
 

os.path.join(path1[, path2[, ...]])  #把目录和文件名合成一个路径
 

os.path.normcase(path)  #转换path的大小写和斜杠
 

os.path.normpath(path)  #规范path字符串形式
 

os.path.realpath(path)  #返回path的真实路径
 

os.path.relpath(path[, start])  #从start开始计算相对路径
 

os.path.samefile(path1, path2)  #判断目录或文件是否相同
 

os.path.sameopenfile(fp1, fp2)  #判断fp1和fp2是否指向同一文件
 

os.path.samestat(stat1, stat2)  #判断stat tuple stat1和stat2是否指向同一个文件
 

os.path.split(path)  #把路径分割成dirname和basename，返回一个元组
 

os.path.splitdrive(path)   #一般用在windows下，返回驱动器名和路径组成的元组
 

os.path.splitext(path)  #分割路径，返回路径名和文件扩展名的元组
 

os.path.splitunc(path)  #把路径分割为加载点与文件
 

os.path.walk(path, visit, arg)  #遍历path，进入每个目录都调用visit函数，visit函数必须有3个参数(arg, dirname, names)，dirname表示当前目录的目录名，names代表当前目录下的所有文件名，args则为walk的第三个参数
 

os.path.supports_unicode_filenames  #设置是否支持unicode路径名
```

## 基本操作

### 文件遍历

```python
import os
rootdir = 'F:\data'
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    if os.path.isfile(path):
        #你想对文件的操作
```

### 文件分割

```python
file_Path = "/home/l/my/test.py"
path,temp = os.path.split(file_Path)
file_name,extension = os.path.splitext(temp)
print('path:',path)
print('file_name:',file_name)
print('extension:',extension)

```

