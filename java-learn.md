---
title: java_learn
categories:
  - 总结
description: <read more ...>
mathjax: true
permalink_defaults: 'category/:title/'
date: 2021-04-25 19:52:00
urlname:
tags:



---

# 基础概念

## .debug

- 如何加断点

  - 选择要设置断点的代码行，在行号的区域后面单击鼠标左键即可

- 如何运行加了断点的程序

  - 在代码区域右键Debug执行

- 看哪里

  - 看Debugger窗口

  - 看Console窗口

- 点哪里

  - 点Step Into (F7)这个箭头，也可以直接按F7

- 如何删除断点

  - 选择要删除的断点，单击鼠标左键即可

  - 如果是多个断点，可以每一个再点击一次。也可以一次性全部删除



##   原码反码补码

1. 原码

   ​	计算机保存最原始的数字，也是没有正和负的数字，叫没符号数字

   ​	左边第一位腾出位置，存放符号，正用0来表示，负用1来表示

2. 反码

   ​	左边第一位腾出位置，存放符号，正用0来表示，负用1来表示

   ​	“反码”表示方式是用来处理负数的，符号位置不变，其余位置相反

   ​	![](java-learn/%E6%AD%A3%E5%8F%8D%E8%A1%A5%E7%A0%81.png)

3. 补码

   ​	解决两个0的问题

   ​	+1,舍弃掉最高位

   ​	![](java-learn/%E6%AD%A3%E5%8F%8D%E8%A1%A5%E7%A0%812.png)

##  位运算

​			亦或与非左右移

# 	方法

##  定义

~~~java
public static void method (    ) {
	// 方法体;
}
public static 返回值类型 方法名(参数) {
   方法体; 
   return 数据 ;
}
~~~

##  调用过程

​		总结：每个方法在被调用执行的时候，都会进入栈内存，并且拥有自己独立的内存空间，方法内部代码调用完毕之后，会从栈内存中弹栈消失。

~~~ java
方法名() 
~~~

##  方法重载

​		overload，同一类中，方法名相同。

##  参数传递

​		基本数据类型的参数，形式参数的改变，不影响实际参数 。

​		原因：每个方法在栈内存中，都会有独立的栈空间，方法运行结束后就会弹栈消失

​		对于引用类型的参数，形式参数的改变，影响实际参数的值 

​		引用数据类型的传参，传入的是地址值，内存中会造成两个引用指向同一个内存的效果，所以即使方法弹栈，堆内存中的数据也已经是改变后的结果 。

# 面向对象

简单理解：类就是对现实事物的一种描述，对象则为具体存在的事物

## 类的定义步骤：**

​	① 定义类

​	② 编写类的成员变量

​	③ 编写类的成员方法

~~~java
public class Student {
    // 属性 : 姓名, 年龄
    // 成员变量: 跟之前定义变量的格式一样, 只不过位置发生了改变, 类中方法外
    String name;
    int age;

    // 行为 : 学习
    // 成员方法: 跟之前定义方法的格式一样, 只不过去掉了static关键字.
    public void study(){
        System.out.println("学习");
    }
}
~~~

## 定义调用

* **创建对象的格式：**
  * 类名 对象名 = new 类名();
* **调用成员的格式：**
  * 对象名.成员变量
  * 对象名.成员方法()

## 对象内存

​			单个对象

​			![单个对象内存图](java-learn/%E5%8D%95%E4%B8%AA%E5%AF%B9%E8%B1%A1%E5%86%85%E5%AD%98%E5%9B%BE.png)

​			多个对象

​			![多个对象内存图](java-learn/%E5%A4%9A%E4%B8%AA%E5%AF%B9%E8%B1%A1%E5%86%85%E5%AD%98%E5%9B%BE.png)

<strong style="color:red;">多个对象在堆内存中，都有不同的内存划分，成员变量存储在各自的内存区域中，成员方法多个对象共用的一份</strong>

​			多个对象指向相同内存图

​			![引用对象内存](java-learn/%E5%BC%95%E7%94%A8%E5%AF%B9%E8%B1%A1%E5%86%85%E5%AD%98.png)

<strong style="color:red;">当多个对象的引用指向同一个内存空间（变量所记录的地址值是一样的）只要有任何一个对象修改了内存中的数据，随后，无论使用哪一个对象进行数据获取，都是修改后的数据</strong>

##  成员变量和局部变量

* **类中位置不同：**成员变量（类中方法外）局部变量（方法内部或方法声明上）
* **内存中位置不同：**成员变量（堆内存）局部变量（栈内存）
* **生命周期不同：**成员变量（随着对象的存在而存在，随着对象的消失而消失）局部变量（随着方法的调用而存在，醉着方法的调用完毕而消失）
* **初始化值不同：**成员变量（有默认初始化值）局部变量（没有默认初始化值，必须先定义，赋值才能使用）

## 封装

### private关键字

特点 :被private修饰的成员，只能在本类进行访问，针对private修饰的成员变量，如果需要被其他类使用，	提供相应的操作

​		提供“get变量名()”方法，用于获取成员变量的值，方法用public修饰

​		提供“set变量名(参数)”方法，用于设置成员变量的值，方法用public修饰

### this关键字

概述 : this修饰的变量用于指代成员变量，其主要作用是（<strong style="color:red;">区分局部变量和成员变量的重名问题</strong>）

* 方法的形参如果与成员变量同名，不带this修饰的变量指的是形参，而不是成员变量
* 方法的形参没有与成员变量同名，不带this修饰的变量指的是成员变量

## 构造方法

* **格式注意 :**
  *  方法名与类名相同，大小写也要一致
  *  没有返回值类型，连void都没有
  *  没有具体的返回值（不能由retrun带回结果数据）
* **执行时机 ：**
  * 创建对象的时候调用，每创建一次对象，就会执行一次构造方法
  * 不能手动调用构造方法
* **执行作用 ：**
  * 用于给对象的数据（属性）进行初始化
* 注意事项  ：
  * 构造方法的注意事项
  * 无论是否使用，都手动书写无参数构造方法，和带参数构造方法

# API

​			API (Application Programming Interface) ：应用程序编程接口

---

##  **如何使用API帮助文档 :** 

​			打开帮助文档

​			找到索引选项卡中的输入框

​			在输入框中输入Random

​			看类在哪个包下

​			看类的描述

​			看构造方法

‘			看成员方法

## 	**Scanner类 :**

​	next() : 遇到了空格, 就不再录入数据了 , 结束标记: 空格, tab键

​	nextLine() : 可以将数据完整的接收过来 , 结束标记: 回车换行符    

##  String类

​	 1 String 类在 java.lang 包下，所以使用的时候不需要导包

​	2 String 类代表字符串，Java 程序中的所有字符串文字（例如“abc”）都被实现为此类的实例也就是说，Java 程序中所有的双引号字符串，都是 String 类的对象

​	3 字符串不可变，它们的值在创建后不能被更改

### 构造方法区别

​	**通过构造方法创建**	通过 new 创建的字符串对象，每一次 new 都会申请一个内存空间，虽然内容相同，但是地址值不同

​	**直接赋值方式创建**	以“”方式给出的字符串，只要字符序列相同(顺序和大小写)，无论在程序代码中出现几次，JVM 都只会建立一个 String 对象，并在字符串池中维护

### 字符串比较， ==使用，equals

- == 比较基本数据类型：比较的是具体的值
- == 比较引用数据类型：比较的是对象地址值
- **String类 :  public boolean equals(String s)     比较两个字符串内容是否相同、区分大小写**

###    字符串截取

​		 public char[] toCharArray( )：

​		substring(0,3);

### 字符串替换

​		 String replace(CharSequence target, CharSequence replacement)

### 切割字符串

```java
String[] split(String regex) ：根据传入的字符串作为规则进行切割将切割后的内容存入字符串数组中，并将字符串数组返回
```
## StringBuilder类概述

**概述 :** StringBuilder 是一个可变的字符串类，我们可以把它看成是一个容器，这里的可变指的是 StringBuilder 对象中的内容是可变的

---

### **常用的构造方法**

| 方法名                             | 说明                                       |
| :--------------------------------- | :----------------------------------------- |
| public StringBuilder()             | 创建一个空白可变字符串对象，不含有任何内容 |
| public StringBuilder(String   str) | 根据字符串的内容，来创建可变字符串对象     |

### StringBuilder常用的成员方法

**添加和反转方法**

| 方法名                                  | 说明                     |
| --------------------------------------- | ------------------------ |
| public StringBuilder   append(任意类型) | 添加数据，并返回对象本身 |
| public StringBuilder   reverse()        | 返回相反的字符序列       |

### StringBuilder和String相互转换

toString,构造方法



## Math类概述

| 方法名    方法名                               | 说明                                           |
| ---------------------------------------------- | ---------------------------------------------- |
| public static int   abs(int a)                 | 返回参数的绝对值                               |
| public static double ceil(double a)            | 返回大于或等于参数的最小double值，等于一个整数 |
| public static double floor(double a)           | 返回小于或等于参数的最大double值，等于一个整数 |
| public   static int round(float a)             | 按照四舍五入返回最接近参数的int                |
| public static int   max(int a,int b)           | 返回两个int值中的较大值                        |
| public   static int min(int a,int b)           | 返回两个int值中的较小值                        |
| public   static double pow (double a,double b) | 返回a的b次幂的值                               |
| public   static double random()                | 返回值为double的正值，[0.0,1.0)                |

## system类



| 方法名                                   | 说明                                             |
| ---------------------------------------- | ------------------------------------------------ |
| public   static void exit(int status)    | 终止当前运行的   Java   虚拟机，非零表示异常终止 |
| public   static long currentTimeMillis() | 返回当前时间(以毫秒为单位)                       |



## Object类概述

### Object类的toString方法（应用）

Object类概述

- Object 是类层次结构的根，每个类都可以将 Object 作为超类。所有类都直接或者间接的继承自该类，换句话说，该类所具备的方法，所有类都会有一份

查看方法源码的方式

- 选中方法，按下Ctrl + B

重写toString方法的方式

- 1. Alt + Insert 选择toString
- 1. 在类的空白区域，右键 -> Generate -> 选择toString

 toString方法的作用：

- 以良好的格式，更方便的展示对象中的属性值

### Object类的equals方法（应用）

equals方法的作用

- 用于对象之间的比较，返回true和false的结果
- 举例：s1.equals(s2);    s1和s2是两个对象

重写equals方法的场景

- 不希望比较对象的地址值，想要结合对象属性进行比较的时候

 重写equals方法的方式

- 1. alt + insert  选择equals() and hashCode()，IntelliJ Default，一路next，finish即可
- 1. 在类的空白区域，右键 -> Generate -> 选择equals() and hashCode()，后面的同上。

### BigDecimal (应用)

​			可以用来进行精确计算

---



| 方法名                 | 说明         |
| ---------------------- | ------------ |
| BigDecimal(double val) | 参数为double |
| BigDecimal(String val) | 参数为String |

| 方法名                                                       | 说明 |
| ------------------------------------------------------------ | ---- |
| public BigDecimal add(另一个BigDecimal对象)                  | 加法 |
| public BigDecimal subtract (另一个BigDecimal对象)            | 减法 |
| public BigDecimal multiply (另一个BigDecimal对象)            | 乘法 |
| public BigDecimal divide (另一个BigDecimal对象)              | 除法 |
| public BigDecimal divide (另一个BigDecimal对象，精确几位，舍入模式) | 除法 |

```java
BigDecimal divide = bd1.divide(参与运算的对象,小数点后精确到多少位,舍入模式);
参数1 ，表示参与运算的BigDecimal 对象。
参数2 ，表示小数点后面精确到多少位
参数3 ，舍入模式  
  BigDecimal.ROUND_UP  进一法
  BigDecimal.ROUND_FLOOR 去尾法
  BigDecimal.ROUND_HALF_UP 四舍五入
```



### 其他常见方法：

| 方法名                                          | 说明                             |
| ----------------------------------------------- | -------------------------------- |
| public static String toString(对象)             | 返回参数中对象的字符串表示形式。 |
| public static String toString(对象, 默认字符串) | 返回对象的字符串表示形式。       |
| public static Boolean isNull(对象)              | 判断对象是否为空                 |
| public static Boolean nonNull(对象)             | 判断对象是否不为空               |

### 

# 集合基础

## ArrayList

**集合和数组的区别 :** 

​	共同点：都是存储数据的容器

​	不同点：数组的容量是固定的，集合的容量是可变的



---

### 构造方法和添加方法

​	

| public ArrayList()                   | 创建一个空的集合对象               |
| ------------------------------------ | ---------------------------------- |
| public boolean add(E e)              | 将指定的元素追加到此集合的末尾     |
| public void add(int index,E element) | 在此集合中的指定位置插入指定的元素 |

**ArrayList<E> ：** 

​	可调整大小的数组实现 

​	<E> : 是一种特殊的数据类型，泛型。

### ArrayList类常用方法

| public boolean remove(Object o)   | 删除指定的元素，返回删除是否成功       |
| --------------------------------- | -------------------------------------- |
| public E remove(int index)        | 删除指定索引处的元素，返回被删除的元素 |
| public E set(int index,E element) | 修改指定索引处的元素，返回被修改的元素 |
| public E get(int index)           | 返回指定索引处的元素                   |
| public int size()                 | 返回集合中的元素的个数                 |

| 方法名                                           | 说明                               |
| ------------------------------------------------ | ---------------------------------- |
| public static String toString(int[] a)           | 返回指定数组的内容的字符串表示形式 |
| public static void sort(int[] a)                 | 按照数字顺序排列指定的数组         |
| public static int binarySearch(int[] a, int key) | 利用二分查找返回指定元素的索引     |

### 常见方法背诵记忆

​		增删改查翻转转换

# git

开发中要解决的问题

+ 代码备份
+ 版本控制
+ 协同工作
+ 责任追溯



---

## 本地添加

![](java-learn/31_Git%E5%9F%BA%E6%9C%AC%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.png)

| 命令                     | 作用                                                   |
| ------------------------ | ------------------------------------------------------ |
| git init                 | 初始化，创建 git 仓库                                  |
| git status               | 查看 git 状态(缓存区) （文件是否进行了添加、提交操作） |
| git add 文件名           | 添加，将指定文件添加到暂存区                           |
| git commit -m '提交信息' | 提交，将暂存区文件提交到历史仓库                       |
| git log                  | 查看日志（ git 提交的历史日志）                        |

## 本地版本管理

+ 准备动作

  1. 查看 my_project 的 log 日志
     git reflog ：可以查看所有分支的所有操作记录（包括已经被删除的 commit 记录的操作）
  2. 增加一次新的修改记录

+ 需求: 将代码切换到第二次修改的版本

  指令：git reset --hard 版本唯一索引值

## 分支管理操作

+ 创建和切换

  创建命令：git branch 分支名
  切换命令：git checkout 分支名

+ 新分支添加文件

  查看文件命令：ls

  总结：不同分支之间的关系是平行的关系，不会相互影响

  即一个分支的操作另一个不受影响

+ 合并分支

  合并命令：git merge 分支名

+ 删除分支

  删除命令：git branch -d 分支名

+ 查看分支列表

  查看命令：git branch

## 先有本地项目,远程为空(应用)

​		 步骤

1. 创建本地仓库

2. 创建或修改文件，添加（add）文件到暂存区，提交（commit）到本地仓库

3. 创建远程仓库

4. 推送到远程仓库

   git remote add 远程名称 远程仓库URL
   git push -u 仓库名称 分支名

## 先有远程仓库,本地为空

1. 将远程仓库的代码，克隆到本地仓库
   克隆命令：git clone 仓库地址
2. 创建新文件，添加并提交到本地仓库
3. 推送至远程仓库
4. 项目拉取更新
   拉取命令：git pull 远程仓库名 分支名

## 代码冲突(应用)

​			情景一：在当前分支上，直接修改冲突代码--->add--->commit。

​			情景二：在本地当前分支上，修改冲突代码--->add--->commit--->push

​			都是通过cat命令查看冲突部分。

​			cat查看冲突文件

# **IDEA集成Git**

## **IDEA中配置Git(应用)**

​		1\. File -\> Settings

![](java-learn/64_IDEA%E4%B8%AD%E9%85%8D%E7%BD%AEGit.png)

2. Version Control -\> Git -\>

指定git.exe存放目录

3\. 点击Test测试

![](java-learn/65_IDEA%E4%B8%AD%E9%85%8D%E7%BD%AEGit.png)

**创建本地仓库(应用)**

>   1\. VCS-\>Import into Version Control-\>Create Git Repository

>   2\. 选择工程所在的目录,这样就创建好本地仓库了

>   3\. 点击git后边的对勾,将当前项目代码提交到本地仓库

>   注意: 项目中的配置文件不需要提交到本地仓库中,提交时,忽略掉即可

![](java-learn/67_%E5%88%9B%E5%BB%BA%E6%9C%AC%E5%9C%B0%E4%BB%93%E5%BA%93.png)

## **版本切换(应用)**

>   方式一: 控制台Version Control-\>Log-\>Reset Current Branch...-\>Reset

>   这种切换的特点是会抛弃原来的提交记录

>   方式二:控制台Version Control-\>Log-\>Revert
>   Commit-\>Merge-\>处理代码-\>commit

>   这种切换的特点是会当成一个新的提交记录,之前的提交记录也都保留

![](java-learn/70_%E7%89%88%E6%9C%AC%E5%88%87%E6%8D%A2.png)

## **分支管理(应用)**

>   创建分支

>   VCS-\>Git-\>Branches-\>New Branch-\>给分支起名字-\>ok

![](java-learn/74_%E5%88%9B%E5%BB%BA%E5%88%86%E6%94%AF.png)

切换分支

idea右下角Git-\>选择要切换的分支-\>checkout

合并分支

VCS-\>Git-\>Merge changes-\>选择要合并的分支-\>merge

![](java-learn/76_%E5%90%88%E5%B9%B6%E5%88%86%E6%94%AF.png)

处理分支中的代码

![](java-learn/78_%E5%90%88%E5%B9%B6%E5%88%86%E6%94%AF.png)

删除分支

idea右下角-\>选中要删除的分支-\>Delete

![](java-learn/80_%E5%88%A0%E9%99%A4%E5%88%86%E6%94%AF.png)

## **本地仓库推送到远程仓库(应用)**

>   1\. VCS-\>Git-\>Push-\>点击master Deﬁne remote

>   2\. 将远程仓库的路径复制过来-\>Push

![](java-learn/81_%E6%9C%AC%E5%9C%B0%E4%BB%93%E5%BA%93%E6%8E%A8%E9%80%81%E5%88%B0%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93.png)

## **远程仓库克隆到本地仓库(应用)**

File->Close Project->Checkout from Version Control->Git->指定远程仓库的路径->指定本地存放的路径->clone

![](java-learn/83_%E8%BF%9C%E7%A8%8B%E4%BB%93%E5%BA%93%E5%85%8B%E9%9A%86%E5%88%B0%E6%9C%AC%E5%9C%B0%E4%BB%93%E5%BA%93.png)

# 分类和static

## 分包思想概述 (理解)

如果将所有的类文件都放在同一个包下,不利于管理和后期维护,所以,对于不同功能的类文件,可以放在不同的包下进行管理

## 包的概述 (记忆)

+ 包

  本质上就是文件夹

+ 创建包

  多级包之间使用 " . " 进行分割
  多级包的定义规范：公司的网站地址翻转(去掉www)
  比如：黑马程序员的网站址为www.itheima.com
  后期我们所定义的包的结构就是：com.itheima.其他的包名

+ 包的命名规则

  字母都是小写

## 包的注意事项 (理解) 

+ package语句必须是程序的第一条可执行的代码 
+ package语句在一个java文件中只能有一个 
+ 如果没有package,默认表示无包名 

## 类与类之间的访问 (理解)

​			同一个包下的访问不需要导包，直接使用即可不同包下的访问

​				1.import 导包后访问

​				2.通过全类名（包名 + 类名）访问注意：import 、package 、class 三个关键字的摆放位置存在顺序关系

					*  package 必须是程序的第一条可执行的代码
	
					*  import 需要写在 package 下面

   * class 需要在 import 下面

     

##  static关键字

### static修饰的特点 (记忆) 

+ 被类的所有对象共享

  是我们判断是否使用静态关键字的条件

+ 随着类的加载而加载，优先于对象存在

  对象需要类被加载后，才能创建

+ 可以通过类名调用

  也可以通过对象名调用

  静态方法只能访问静态的成员

  非静态方法可以访问静态的成员，也可以访问非静态的成员

# 数组的高级操作 

## 二分查找

```java
public class MyBinarySearchDemo {
    public static void main(String[] args) {
        int [] arr = {1,2,3,4,5,6,7,8,9,10};
        int number = 11;

        //1,我现在要干嘛? --- 二分查找
        //2.我干这件事情需要什么? --- 数组 元素
        //3,我干完了,要不要把结果返回调用者 --- 把索引返回给调用者
        int index = binarySearchForIndex(arr,number);
        System.out.println(index);
    }

    private static int binarySearchForIndex(int[] arr, int number) {
        //1,定义查找的范围
        int min = 0;
        int max = arr.length - 1;
        //2.循环查找 min <= max
        while(min <= max){
            //3.计算出中间位置 mid
            int mid = (min + max) >> 1;
            //mid指向的元素 > number
            if(arr[mid] > number){
                //表示要查找的元素在左边.
                max = mid -1;
            }else if(arr[mid] < number){
                //mid指向的元素 < number
                //表示要查找的元素在右边.
                min = mid + 1;
            }else{
                //mid指向的元素 == number
                return mid;
            }
        }
        //如果min大于了max就表示元素不存在,返回-1.
        return -1;
    }
  
}
```

## 冒泡排序

一种排序的方式，对要进行排序的数据中相邻的数据进行两两比较，将较大的数据放在后面，依次对所有的数据进行操作，直至所有数据按要求完成排序

如果有n个数据进行排序，总共需要比较n-1次

每一次比较完毕，下一次的比较就会少一个数据参与

```java
public class MyBubbleSortDemo2 {
    public static void main(String[] args) {
        int[] arr = {3, 5, 2, 1, 4};
        //1 2 3 4 5
        bubbleSort(arr);
    }

    private static void bubbleSort(int[] arr) {
        //外层循环控制的是次数 比数组的长度少一次.
        for (int i = 0; i < arr.length -1; i++) {
            //内存循环就是实际循环比较的
            //-1 是为了让数组不要越界
            //-i 每一轮结束之后,我们就会少比一个数字.
            for (int j = 0; j < arr.length - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }

        printArr(arr);
    }

    private static void printArr(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }
  
}
```

## 快速排序 (理解)

+ 快速排序概述

  冒泡排序算法中,一次循环结束,就相当于确定了当前的最大值,也能确定最大值在数组中应存入的位置

  快速排序算法中,每一次递归时以第一个数为基准数,找到数组中所有比基准数小的.再找到所有比基准数大的.小的全部放左边,大的全部放右边,确定基准数的正确位置

+ 核心步骤

  1. 从右开始找比基准数小的
  2. 从左开始找比基准数大的
  3. 交换两个值的位置
  4. 红色继续往左找，蓝色继续往右找，直到两个箭头指向同一个索引为止
  5. 基准数归位

```java
public class MyQuiteSortDemo2 {
    public static void main(String[] args) {
//        1，从右开始找比基准数小的
//        2，从左开始找比基准数大的
//        3，交换两个值的位置
//        4，红色继续往左找，蓝色继续往右找，直到两个箭头指向同一个索引为止
//        5，基准数归位
        int[] arr = {6, 1, 2, 7, 9, 3, 4, 5, 10, 8};

        quiteSort(arr,0,arr.length-1);

        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
    }

    private static void quiteSort(int[] arr, int left, int right) {
     	// 递归结束的条件
        if(right < left){
            return;
        }

        int left0 = left;
        int right0 = right;

        //计算出基准数
        int baseNumber = arr[left0];

        while(left != right){
//        1，从右开始找比基准数小的
            while(arr[right] >= baseNumber && right > left){
                right--;
            }
//        2，从左开始找比基准数大的
            while(arr[left] <= baseNumber && right > left){
                left++;
            }
//        3，交换两个值的位置
            int temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
        }
        //基准数归位
        int temp = arr[left];
        arr[left] = arr[left0];
        arr[left0] = temp;
      
		// 递归调用自己,将左半部分排好序
        quiteSort(arr,left0,left-1);
      	// 递归调用自己,将右半部分排好序
        quiteSort(arr,left +1,right0);

    }
}
```

# 异常

## 异常的体系结构

![](java-learn/01_%E5%BC%82%E5%B8%B8%E4%BD%93%E7%B3%BB%E7%BB%93%E6%9E%84.png)

## 编译时异常和运行时异常的区别

![](java-learn/02_%E7%BC%96%E8%AF%91%E6%97%B6%E5%BC%82%E5%B8%B8%E5%92%8C%E8%BF%90%E8%A1%8C%E6%97%B6%E5%BC%82%E5%B8%B8.png)

- 编译时异常

  - 都是Exception类及其子类
  - 必须显示处理，否则程序就会发生错误，无法通过编译

- 运行时异常

  - 都是RuntimeException类及其子类
  - 无需显示处理，也可以和编译时异常一样处理

  ![](java-learn/02_%E7%BC%96%E8%AF%91%E6%97%B6%E5%BC%82%E5%B8%B8%E5%92%8C%E8%BF%90%E8%A1%8C%E6%97%B6%E5%BC%82%E5%B8%B8.png)

## 查看异常信息 (理解) 

控制台在打印异常信息时,会打印异常类名,异常出现的原因,异常出现的位置

我们调bug时,可以根据提示,找到异常出现的位置,分析原因,修改异常代码

![](java-learn/03_%E6%9F%A5%E7%9C%8B%E5%BC%82%E5%B8%B8%E4%BF%A1%E6%81%AF.png)

## throws方式处理异常

```java
public void 方法() throws 异常类名 {
    
}
```

​		

```java
public class ExceptionDemo {
    public static void main(String[] args) throws ParseException{
        System.out.println("开始");
//        method();
          method2();

        System.out.println("结束");
    }

    //编译时异常
    public static void method2() throws ParseException {
        String s = "2048-08-09";
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        Date d = sdf.parse(s);
        System.out.println(d);
    }

    //运行时异常
    public static void method() throws ArrayIndexOutOfBoundsException {
        int[] arr = {1, 2, 3};
        System.out.println(arr[3]);
    }
}
```

# Optional

Optional 类主要解决的问题是臭名昭著的空指针异常（NullPointerException）

---

| 方法名                                     | 说明                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| static <T> Optional<T> of(T value)         | 获取一个Optional对象，封装的是非null值的对象                 |
| static <T> Optional<T> ofNullable(T value) | 获取一个Optional对象，Optional封装的值对象可以是null也可以不是null |

```java
public class OptionalDemo1 {
    public static void main(String[] args) {
        //method1();

        //public static <T> Optional<T> ofNullable(T value)
        //获取一个Optional对象，Optional封装的值对象可以是null也可以不是null
        //Student s = new Student("zhangsan",23);
        Student s = null;
        //ofNullable方法，封装的对象可以是null，也可以不是null。
        Optional<Student> optional = Optional.ofNullable(s);

        System.out.println(optional);
    }

    private static void method1() {
        //static <T> Optional<T> of(T value)    获取一个Optional对象，封装的是非null值的对象

        //Student s = new Student("zhangsan",23);
        Student s = null;
        //Optional可以看做是一个容器，里面装了一个引用数据类型的对象。
        //返回值就是Optional的对象
        //如果使用of方法，封装的对象如果为空，那么还是会抛出空指针异常
        Optional<Student> optional1 = Optional.of(s);
        System.out.println(optional1);
    }
}
```

```java
import java.util.Optional;
 
public class Java8Tester {
   public static void main(String args[]){
   
      Java8Tester java8Tester = new Java8Tester();
      Integer value1 = null;
      Integer value2 = new Integer(10);
        
      // Optional.ofNullable - 允许传递为 null 参数
      Optional<Integer> a = Optional.ofNullable(value1);
        
      // Optional.of - 如果传递的参数是 null，抛出异常 NullPointerException
      Optional<Integer> b = Optional.of(value2);
      System.out.println(java8Tester.sum(a,b));
   }
    
   public Integer sum(Optional<Integer> a, Optional<Integer> b){
    
      // Optional.isPresent - 判断值是否存在
        
      System.out.println("第一个参数值存在: " + a.isPresent());
      System.out.println("第二个参数值存在: " + b.isPresent());
        
      // Optional.orElse - 如果值存在，返回它，否则返回默认值
      Integer value1 = a.orElse(new Integer(0));
        
      //Optional.get - 获取值，值需要存在
      Integer value2 = b.get();
      return value1 + value2;
   }
}
$ javac Java8Tester.java 
$ java Java8Tester
第一个参数值存在: false
第二个参数值存在: true
10
```

# 继承

## 继承的实现

- 继承的概念

  - 继承是面向对象三大特征之一，可以使得子类具有父类的属性和方法，还可以在子类中重新定义，以及追加属性和方法

- 实现继承的格式

  - 继承通过extends实现
  - 格式：class 子类 extends 父类 { } 
    - 举例：class Dog extends Animal { }

- 继承带来的好处

  - 继承可以让类与类之间产生关系，子父类关系，产生子父类后，子类则可以使用父类中非私有的成员。

## 继承的好处和弊端（理解）

- 继承好处
  - 提高了代码的复用性(多个类相同的成员可以放到同一个类中)
  - 提高了代码的维护性(如果方法的代码需要修改，修改一处即可)
- 继承弊端
  - 继承让类与类之间产生了关系，类的耦合性增强了，当父类发生变化时子类实现也不得不跟着变化，削弱了子类的独立性
- 继承的应用场景：
  - 使用继承，需要考虑类与类之间是否存在is..a的关系，不能盲目使用继承
    - is..a的关系：谁是谁的一种，例如：老师和学生是人的一种，那人就是父类，学生和老师就是子类

## Java中继承,多继承和多层继承（掌握）

- Java中继承的特点

  1. Java中类只支持单继承，不支持多继承
     - 错误范例：class A extends B, C { }
  2. Java中类支持多层继承

继承中的成员访问特点

##  继承中变量的访问特点（掌握）

在子类方法中访问一个变量，采用的是就近原则。

1. 子类局部范围找
2. 子类成员范围找
3. 父类成员范围找
4. 如果都没有就报错(不考虑父亲的父亲…)

## super（掌握）

- this&super关键字：
  - this：代表本类对象的引用
  - super：代表父类存储空间的标识(可以理解为父类对象引用)
- this和super的使用分别
  - 成员变量：
    - this.成员变量    -   访问本类成员变量
    - super.成员变量 -   访问父类成员变量
  - 成员方法：
    - this.成员方法  - 访问本类成员方法
    - super.成员方法 - 访问父类成员方法
- 构造方法：
  - this(…)  -  访问本类构造方法
  - super(…)  -  访问父类构造方法
  
  ![](java-learn/01_super%E5%86%85%E5%AD%98%E5%9B%BE.png)

## 方法重写

- 1、方法重写概念

  - 子类出现了和父类中一模一样的方法声明（方法名一样，参数列表也必须一样）

- 2、方法重写的应用场景

  - 当子类需要父类的功能，而功能主体子类有自己特有内容时，可以重写父类中的方法，这样，即沿袭了父类的功能，又定义了子类特有的内容

- 3、Override注解

  - 用来检测当前的方法，是否是重写的方法，起到【校验】的作用

  方法重写的注意事项

  1. 私有方法不能被重写(父类私有成员子类是不能继承的)
  2. 子类方法访问权限不能更低(public > 默认 > 私有)
  3. 静态方法不能被重写,如果子类也有相同的方法,并不是重写的父类的方法

## 权限修饰符

![](java-learn/02_%E6%9D%83%E9%99%90%E4%BF%AE%E9%A5%B0%E7%AC%A6.png)