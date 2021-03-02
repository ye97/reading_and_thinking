---
title: git
date: 2021-01-21 19:34:05
tags:
  - git
categories:
  - git

---



1.新建项目

git init

或者

git clone

2.将项目和远端连接

git remote add origin main

3.将本地修改文件添加到commit区

git add .

git commit -m "message"

(可以使用git status来查看提交状态)

4.推送到远端

将本地分支和远程分支连接起来

git push -u origin mybranch1 

git branch --set-upstream-to=origin/mybranch1 mybranch1

一般使用第一种即可表示同名分支之间的连接

5.将本地推送到远程分支

​    5.1 doesnt match anything 表示本地分支的提交区没有东西，或者没有远程分支对应，情况一：牢记先提交后push

情况二：git remote 查看当前分支所联系的远程是否正确，

  5.2版本不一致：

   git fetch 拉取下来，修改冲突之后合并

2，强制push，只要自己确定本地仓库版本更好

3.使用stash

常规 git stash 的一个限制是它会一下暂存所有的文件。有时，只备份某些文件更为方便，让另外一些与代码库保持一致。一个非常有用的技巧，用来备份部分文件：

add 那些你不想备份的文件（例如： git add file1.js, file2.js）
调用 git stash –keep-index。只会备份那些没有被add的文件。
调用 git reset 取消已经add的文件的备份，继续自己的工作。



