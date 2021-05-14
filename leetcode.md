---
title: leetcode
categories:
  - leetcode
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-05-13 21:21:45
urlname:
tags:
---











# 滑动窗口

```c++
string minWindow(string s, string t) {
    int left=0,right=0;
    int valid=0;
    unordered_map<char,int> need,window;
    for(auto i:t) need[i]++;
    while(right<s.size()){
        //不论如何窗口都是要一直增大的
            char c=s[right];
        // 进行窗口内数据的一系列更新

            right++;

        while(//window needs shrink){
            char d=s[left];
             //更新结果
             //更新窗口（进行窗口内数据的一系列更新）
            left++;

        }


    }

    return //结果;
}
```

> ​		总结：
>
> ​		1，先创建一个windows窗口，存放所遍历的元素
>
> ​		2，need是辅助数组，一般用于第二个while的缩小判断
>
> ​		3， 先取右端元素
>
> ​				更新窗口和辅助数组的值
>
> ​				判断是否需要缩小
>
> ​				取左端元素（一个一个的缩小）
>
> ​				更新窗口和辅助数组的值
>
> ​				right和left更新都是最后更新
>
> ​		<strong style="color:red;">		所需结果只能是缩小窗口时或者下一个稳定态时取到，都是比较值</strong>

## 举例：

