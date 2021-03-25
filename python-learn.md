---
title: python_learn
categories:
  - python
tags:
  - 笔记
  - 总结
  - 语法
date: 2021-03-04 22:11:24

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

## 1，cumsum

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

## 2，np.concatenate

~~~python
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[11,21,31],[7,8,9]])
print(np.concatenate((a,b),axis=1))
#[[ 1  2  3 11 21 31]
#[ 4  5  6  7  8  9]]
~~~

## 3，np.zeros

## 4， np.sort

## 5，**numpy.arange([start, ]stop, [step, ]dtype=None)**

在给定的时间间隔内返回均匀间隔的值。



