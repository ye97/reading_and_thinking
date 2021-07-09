---
title: 如何让markdown自动显示序号
categories:
  - 编程工具
  - hexo
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-05-19 00:45:30
urlname:
tags:
typora-root-url: 如何让markdown自动显示序号
---



# \[Markdown\]\[typora\]如何让markdown自动显示序号



## 介绍

我们经常会遇到在写markdown的时候，需要显示标题。这个时候，通常大家的做法是手动添加标题，但是写到最后会发现要在添加一个之前的章节，结果后面的写好的都要跟着改，那markdown有没有办法自动显示标题呢？这边我搜索了相关博客，学习了一下。记录下来。

### 第一步

打开typora的外观里面的`主题文件夹`  
![在这里插入图片描述](/1621355838-88d5d01a6004b6e8915160d4cdd8a964.gif)

### 第二步：

新建一个名为base.user.css的文件

![image-20210705224716816](/image-20210705224716816.png)

### 第三步：

添加如下内容即可：  
上边那部分是左边的标题栏，下面那部分是正文的标题

```
#write {
    counter-reset: h1
}

h1 {
    counter-reset: h2
}

h2 {
    counter-reset: h3
}

h3 {
    counter-reset: h4
}

h4 {
    counter-reset: h5
}

h5 {
    counter-reset: h6
}

#write h1:before {
    counter-increment: h1;
    content: counter(h1) " "
}

#write h2:before {
    counter-increment: h2;
    content: counter(h1) "." counter(h2) " "
}

#write h3:before,
h3.md-focus.md-heading:before {
    counter-increment: h3;
    content: counter(h1) "." counter(h2) "." counter(h3) " "
}

#write h4:before,
h4.md-focus.md-heading:before {
    counter-increment: h4;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) " "
}

#write h5:before,
h5.md-focus.md-heading:before {
    counter-increment: h5;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) "." counter(h5) " "
}

#write h6:before,
h6.md-focus.md-heading:before {
    counter-increment: h6;
    content: counter(h1) "." counter(h2) "." counter(h3) "." counter(h4) "." counter(h5) "." counter(h6) " "
}

#write > h3.md-focus:before,
#write > h4.md-focus:before,
#write > h5.md-focus:before,
#write > h6.md-focus:before,
h3.md-focus:before,
h4.md-focus:before,
h5.md-focus:before,
h6.md-focus:before {
    color: inherit;
    border: inherit;
    border-radius: inherit;
    position: inherit;
    left: initial;
    float: none;
    top: initial;
    font-size: inherit;
    padding-left: inherit;
    padding-right: inherit;
    vertical-align: inherit;
    font-weight: inherit;
    line-height: inherit;
}
```

### 第四步：

看下效果：  
![在这里插入图片描述](1621355838-59090767a22d5732212c3e3e1f1553fc.png)