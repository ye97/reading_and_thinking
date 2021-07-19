---
title: 如何让markdown自动显示序号
categories:
  - 开发
  - markdown
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-05-19 00:45:30
urlname:
tags:
typora-root-url: 如何让markdown自动显示序号
---



# 自动显示序号

## 第一步

打开typora的外观里面的`主题文件夹`  
![在这里插入图片描述](/../markdown/1625886167-88d5d01a6004b6e8915160d4cdd8a964.gif)

## 第二步：

新建一个名为base.user.css的文件

## 第三步：

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

## 第四步：

看下效果：  
![在这里插入图片描述](/../markdown/1625886167-59090767a22d5732212c3e3e1f1553fc.png)

# 语法总结

注意：很多需要效果展示的地方，参考配图中的红色区域。

1\. 标题

第一种是使用 `#` 表示标题，其中 `#` 号必须在行首，  
第二种是使用 `===` 或者 `---` 表示。

![](/../markdown/1625885851-873b15a7e4ad983a89f1d274e8c30a0f.png)

2\. 分割线

使用三个或以上的 `-` 或者 `*` 表示，且这一行只有符号，**注意不要被识别为二级标题即可**，例如中间或者前面可以加空格。

3\. 斜体和粗体

使用 `*` 和 `**` 分别表示斜体和粗体，删除线使用两个 `~` 表示

![](/../markdown/1625885851-c1eb3ba5edcf11b9ae6c48d848aca819.png)

4\. 超链接和图片

链接和图片的写法类似，图片仅在超链接前多了一个 `!` ，一般是 \[文字描述\] (链接)

5\. 无序列表

使用 `-`、`+` 和 `*` 表示无序列表，前后留一行空白，可嵌套，例如

![](/../markdown/1625885851-a66415ba457f13ea818c385e83e1d90f.png)

6\. 有序列表

使用 `1.` （点号后面有个空格）表示有序列表，可嵌套。

7\. 文字引用

使用 `>` 表示，可以有多个 `>`，表示层级更深，例如

![](/../markdown/1625885851-6c1a03d6783e85ecdd8d8c7076d74182.png)

8\. 行内代码块

其实上面已经用过很多次了，即使用 \` 表示，例如

扩展：很多字符是需要转义，使用反斜杠 `\` 进行转义

9\. 代码块

使用四个空格缩进表示代码块，一些 IDE 支持行数提示和着色，一般使用三个 \` 表示，例如

![](/../markdown/1625885851-82b59aa129437eebd017add5a67c26bd.png)

10\. 表格

直接看例子吧，第二行的 `---:` 表示了对齐方式，默认**左对齐**，还有 **右对齐** 和 **居中**

| 商品 | 数量 |  单价   |
| ---- | ---: | :-----: |
| 苹果 |   10 |  \\$1   |
| 电脑 |    1 | \\$1000 |

**![](/../markdown/1625885851-2ab0503a5449a8123ba8275ca65c27cc.png)**

11\. 流程图

主要的语法为 `name=>type: describe`，其中 type 主要有以下几种：  
1.开始和结束：`start` `end`  
2.输入输出：`inputoutput`  
3.操作：`operation`  
4.条件：`condition`  
5.子程序：`subroutine`

![](/../markdown/1625885851-fa003fc7f0c69a0dd0050baf396a3c58.png)

更多语法参考：[流程图语法参考](http://adrai.github.io/flowchart.js/)

12\. 数学公式

使用 `$` 表示，其中一个 $ 表示在行内，两个 $ 表示独占一行。

eg : $\\sum\_{i=1}^n a\_i=0$

支持 **LaTeX** 编辑显示支持，访问 [MathJax](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference) 参考更多使用方法。

推荐一个常用的数学公式在线编译网站：[https://private.codecogs.com/latex/eqneditor.php](https://private.codecogs.com/latex/eqneditor.php)

13.支持 HTML 标签

例如想要段落的缩进，可以如下：

&nbsp;&nbsp;不断行的空白格&nbsp;或&#160;  
&ensp;&ensp;半方大的空白&ensp;或&#8194;  
&emsp;&emsp;全方大的空白&emsp;或&#8195;

![](/../markdown/1625885851-f781184aaa3c8145afaa0c70da5c520b.png)

点我跳转的功能这里演示不了，写法如下：

    <h6 id='anchor'>我是一个锚点</h6>

    \[点我跳转\](#anchor)

