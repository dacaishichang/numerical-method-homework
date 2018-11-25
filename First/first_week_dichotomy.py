# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:28:22 2018
二分法
@author: hhuaf
"""
import numpy as np
import matplotlib.pyplot as plt
# input
'''
a :下限
b :上限
theta:阈值
'''
#a=input('输入下限\n:')
#b=input('输入上限\n:')
#a=float(a)
#b=float(b)
a=1
b=5
theta=0.05

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')

# 定义函数，构造数值

fun = lambda x: x**3-3*x+1
#fun = lambda x: x**2-1
x = np.arange(a,b,0.05)
y = fun(x)

# 画出图像

fig = plt.figure(figsize = (15, 10))
plt.subplot(2,3,1)
plt.xlim(a, b)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('$f(x)=x^3-3x+1$ 图像')
plt.plot(x,y)


# 记录x0的收敛情况
list_x=[]

# 记录|a-b|的收敛情况
abs_list=[]

# 定义二分求解算法
# 判断 a b是否合法 a<b ab<0 theta
def dichotomy(func = fun,a = -1,b = 1,theta=0.05):
    number=0
    try:
        #数值错误
        if a>b or func(a)*func(b)>0 or abs(a-b) < theta:
            raise ValueError

        #有解
        if func(a)*func(b) == 0:
				
            if func(a) == 0:
                return a
            else:
                return b
        
        while True:
            number+=1
            c = (a+b)/2
        
            list_x.append(c)
            abs_list.append(abs(a-b))
            
            if func(c) == 0:
                
                return c
            else: 
                if func(a) * func(c) < 0:
                    b=c
                else:
                    a=c
                if abs(a-b)<theta:
                    
                    return c
            if number >= 200:
                print("给定范围内可能有解，但是无法收敛！")
                return None
    except ValueError:
        print("边界不对或者范围不对！")
        return None

# 计算求解x0 

x0=dichotomy(fun, a, b, theta)
if x0 != None:
    print("求解结果为：x0 = ",x0)
    print("f(x0) = ",fun(x0))
# 求解图像
plt.subplot(2,3,2)
plt.xlim(a, b)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('求解结果')
plt.plot(x,y)

text = "二分法求得点"
if x0 != None:
    plt.scatter(x0,fun(x0),c='black')
    
else:
    plt.title('区间上无解')
     
# 查看解的收敛情况x
plt.subplot(2,3,3)
plt.xlabel('次数')
plt.ylabel('x0')
plt.title('x0的收敛情况')
plt.plot(range(len(list_x)),list_x)


# 查看f(x)的收敛情况
plt.subplot(2,3,4)
plt.xlabel('次数')
plt.ylabel('f(x)')
plt.title('f(x)的收敛情况')
plt.plot(range(len(list_x)),[fun(x) for x in list_x])

#查看|a-b|
plt.subplot(2,3,5)
plt.xlabel('次数')
plt.ylabel('|a-b|')
plt.title('|a-b|收敛情况')
plt.plot(range(len(list_x)),abs_list)
plt.show()