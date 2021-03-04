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

