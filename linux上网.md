---
title: linux上网
categories:
  - 开发
  - 编程工具
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-07-16 09:42:29
urlname:
tags:
---

# VMware虚拟机中ubuntu无法联网问题的解决方法

**![VMware虚拟机中ubuntu无法联网问题的解决方法（亲测有效）](linux%E4%B8%8A%E7%BD%91/1626399675-868dcff53804025e9277b6cc55415977.ico)摘要：**1、保证自己的电脑能正常连接网络； 2、打开关于VMware的所有服务（一般情况服务设置的是手动启动，需要自己打开）； 3、对VMware虚拟机进行网络设置； 4、选择网络适配器选项，将连接方式设置成NAT模式，并进行保存； 5、将本机网络进行共享，并保存； 6、打开VMware进入ubuntu系统，网络已正常连接并能正常上网。

1、保证自己的电脑能正常连接网络；

2、打开关于VMware的所有服务（一般情况服务设置的是手动启动，需要自己打开）如图：

![VMware虚拟机中ubuntu无法联网问题的解决方法](linux%E4%B8%8A%E7%BD%91/1626399675-5af31301aca1f1ef45fe9c6f32ec4ffe.jpg "VMware虚拟机中ubuntu无法联网问题的解决方法")

  

3、对VMware虚拟机进行网络设置：右击ubuntn选择设置

![VMware虚拟机中ubuntu无法联网问题的解决方法](linux%E4%B8%8A%E7%BD%91/1626399675-9bae5880e953c14d17be343db1d05fdb.jpg "VMware虚拟机中ubuntu无法联网问题的解决方法")

4、选择网络适配器选项，将连接方式设置成NAT模式，并进行保存。

  

![VMware虚拟机中ubuntu无法联网问题的解决方法](linux%E4%B8%8A%E7%BD%91/1626399675-3d0a2d074349e1bd016f8d79b0acb844.jpg "VMware虚拟机中ubuntu无法联网问题的解决方法")

  

5、将本机网络进行共享，并保存；

![VMware虚拟机中ubuntu无法联网问题的解决方法](linux%E4%B8%8A%E7%BD%91/1626399675-9ebfefb920b573330a606776642c22e2.jpg "VMware虚拟机中ubuntu无法联网问题的解决方法")

6、打开VMware进入ubuntu系统，网络已正常连接并能正常上网。

![](linux%E4%B8%8A%E7%BD%91/Snipaste_2021-07-16_09-44-57.png)

