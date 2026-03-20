# 这是关于numpy用法的笔记



**创建array**

``` python
import numpy as np

array = np.array([[1,2,3],[2,3,4]])
a = np.zeros((3,4))  # 几行几列用括号括住 生成三行四列的零矩阵

a = np.ones((3,4),dype = np.int16)

a = np.arange(10, 20，2) # 生成[10,12,14,16,18]

a = np.linspace(1, 10, 5) # 自动匹配步长

print(array)  # 输出arrray矩阵

print('number of dim:',array.ndim) #数组维度

print('shape:',array.shape) #(行，列)

print('size:',array.size) #总共有多少个元素
```



**定义类型**

```python
a = np.array([2,23,4],dtype = np.int) 
# int可以替换为其他类型，位数越小占用空间越小，比较精确：float64
```



**基础运算**

```
a = np.array([10,20,30,40])
b = np.arange(4)

print(np.mean(A)) # 求平均值
print(np.median(A))  #求中位数
print(np.cumsum(A))  #累加（前n项和）
print(np.diff(A))  # 后一个减去前一个的差
print(np.nonzero)  # 输出非零数的索引（用两个array表示）
print(np.sort(A))  # 逐行升序排序


```



##### 矩阵相关

```python
a = np.array([[1,1],[0,1]])
b = np.arange(4).reshape((2,2))

c = a*b  # 逐个相乘
c_dot = np.dot(a,b)   # 矩阵乘法  点积  计算投影和夹角
c_cross = np.cross(a,b)  # 叉积  计算平行四边形面积（二维时）

sum = np.sum(a)
sum_2 = np.sum(a,axis = 1) # 第一行的和
min = np.min(a,axis = 0)  # 按列求最小值，axis = 1按行

A = np.arange(2,14)reshape((2,14))
print(np.argmin(A)) #  求最小值索引 argmax(A)最大值索引


print(np.transpose(A))  # j矩阵的转置

np.linalg.det()  # 行列式的绝对值，计算面积/体积  判断线性无关

np.linalg.norm()  # 求向量的模长
np.linalg.matrix.rank()  # 判断线性无关
```



##### 解方程

法（一）

```
np.linalg.solve() 
```

法（二）

```
B_inv = np.linalg.inv(B)
new_v = (B_inv @ v).T
```



**合法的坐标系条件**

==线性无关==

用矩阵的秩或行列式的值判断

 ```python
 np.linalg.matrix_rank(A)  # numpy计算秩的用法
 np.isclose(det_A,0)  # 看行列式的值是否近似于零
 ```

##### 关于角度

```python
np.degree()  # 用于弧度转角度
np.nan  # 不是一个数的表示
np.sin(a)  #对每个元素取sin值再乘10
np.clip(number,min,max)  
# 强制将一个数限制在一个范围之内，用于消除浮点精度不足的问题，防止余弦值溢出
```







