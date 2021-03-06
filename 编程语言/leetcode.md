---
title: leetcode
categories:
  - 开发
  - java
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

### 最短超串



> 假设你有两个数组，一个长一个短，短的元素均不相同。找到长数组中包含短数组所有的元素的最短子数组，其出现顺序无关紧要。
>
> 返回最短子数组的左端点和右端点，如有多个满足条件的子数组，返回左端点最小的一个。若不存在，返回空数组。
>
> **示例 1:**

> ```
> 输入:
> big = [7,5,9,0,2,1,3,5,7,9,1,1,5,8,8,9,7]
> small = [1,5,9]
> 输出: [7,10]
> ```

> **示例 2:**

> ```
> 输入:
> big = [1,2,3]
> small = [4]
> 输出: []
> ```





```java
class Solution {
public:
    vector<int> shortestSeq(vector<int>& big, vector<int>& small) {
        vector<int> ans;
        /*窗口m*/
        unordered_map<int,int> m;
         /*count限制因素*/
        int count=0,j=0;
        for(auto x:small){
            if(!m.count(x)) count++;
            m[x]++;
        } 
        for(int i=0;i<big.size();i++){
            /*取右窗口*/
            m[big[i]]--;
            /*更新窗口值，限制值*/
            if(m[big[i]]==0) count--;
            /*更新窗口条件*/
            while(!count){
                
                m[big[j]]++;
                if(m[big[j]]>0){
                    count++;
                    if(ans.empty()||ans[1]-ans[0]>i-j) ans={j,i};
                } 
                /*最后更新j*/
                j++;
            }
        }
        return ans;
    }
};
```

### [最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

> 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
>
> 注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

**示例 1：**

> ```
> 输入：s = "ADOBECODEBANC", t = "ABC"
> 输出："BANC"
> ```

**示例 2：**

> ```
> 输入：s = "a", t = "a"
> 输出："a"
> ```

```java
class Solution {
    public String minWindow(String s, String t) {
     if (s==null||t==null||s.length()<t.length()) return "";

        char[] sArr=s.toCharArray();
        char[] tArr=t.toCharArray();
        int[] hash=new int[128];
        for (char c:tArr) hash[c]++;
        int l=0,count=t.length(),max=s.length()+1;
        int r=0;
        String result="";
        while(r<sArr.length){
            hash[sArr[r]]--;
            if (hash[sArr[r]]>=0){
                count--;
            }
            while (r>l&&hash[sArr[l]]<0){
                hash[sArr[l]]++;
                l++;
            }
            if (count==0&&max>r-l+1){
                max=r-l+1;
                result=s.substring(l,r+1);
            }
            r++;
        }
        
         return result;
 }
}
```

### 无重复字符的最长子串

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

**示例 1:**

```plain
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```plain
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```plain
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

**示例 4:**

```plain
输入: s = ""
输出: 0
```

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.length()==0) return 0;
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for(int i = 0; i < s.length(); i ++){
            /*此处使用if而不是while，因为可以确定只需要排除一位即可*/
            if(map.containsKey(s.charAt(i))){
                left = Math.max(left,map.get(s.charAt(i)) + 1);
            }
            /*更新权重*/
            map.put(s.charAt(i),i);
            max = Math.max(max,i-left+1);
        }
        return max;
    }
}


```

### [至少有 K 个重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/)

给你一个字符串 `s` 和一个整数 `k` ，请你找出 `s` 中的最长子串， 要求该子串中的每一字符出现次数都不少于 `k` 。返回这一子串的长度。

**示例 1：**

```plain
输入：s = "aaabb", k = 3
输出：3
解释：最长子串为 "aaa" ，其中 'a' 重复了 3 次。
```

**示例 2：**

```plain
输入：s = "ababbc", k = 2
输出：5
解释：最长子串为 "ababb" ，其中 'a' 重复了 2 次， 'b' 重复了 3 次。
```

**提示：**

*   `1 <= s.length <= 104`
*   `s` 仅由小写英文字母组成
*   `1 <= k <= 105



```JAVA
class Solution {
    public int longestSubstring(String s, int k) {
        int ret = 0;
        int n = s.length();
        for (int t = 1; t <= 26; t++) {
            //用t枚举所有可能的字符数两
            //当t固定，对这个情况进行滑窗
            int l = 0, r = 0;
            int[] cnt = new int[26];
            //cnt 是记录所有字符
            int tot = 0;
            //tot是窗口所有字符种数
            int less = 0;
            //一个计数器 less，代表当前出现次数小于 kk 的字符的数量
            while (r < n) {
                cnt[s.charAt(r) - 'a']++;
                if (cnt[s.charAt(r) - 'a'] == 1) {
                    tot++;
                    less++;
                }
                if (cnt[s.charAt(r) - 'a'] == k) {
                    less--;
                }

                while (tot > t) {
                    cnt[s.charAt(l) - 'a']--;
                    if (cnt[s.charAt(l) - 'a'] == k - 1) {
                        less++;
                    }
                    if (cnt[s.charAt(l) - 'a'] == 0) {
                        tot--;
                        less--;
                    }
                    l++;
                }
                if (less == 0) {
                    ret = Math.max(ret, r - l + 1);
                }
                r++;
            }
        }
        return ret;
    }
}
```



