# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:56:58 2018
劈因子法（贝尔斯托法）
@author: hhuaf
"""
import numpy as np
import matplotlib.pyplot as plt

#input_u=input('请输入两个数，第一个')
#iuput_v=input('请输入两个数，第二个')
input_u=0
iuput_v=0

theta=1e-5
num=1000

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')


# 因子式
def f(x):
    return (x+1)*(x-4)*(x-5)*(x+3)*(x-2)
   
# 一般式
def f1(x):
    return x**5-7*x**4-3*x**3+79*x**2-46*x-120
# 原函数系数

coefficient=[1,-7,-3,79,-46,-120]
root_list=[]

def two(a):
    return -a[1]/a[0]

def Bairstow(a = coefficient, u = 0, v = 0, num = 1000, theta = 1e-2 ):
    if len(a) < 2:
        return [None]
    if len(a) == 2:
        return [two(a)]
    
    elif len(a) == 3:
        delta_num = (u**2-4*v)
        x1=0
        x2=0
        if (delta_num < 0):
            x1=complex(-u/2,np.sqrt(abs(delta_num))/2)
            x2=complex(-u/2,-np.sqrt(abs(delta_num))/2)
        else:
            x1 = -u/2+np.sqrt(delta_num)/2
            x2 = -u/2-np.sqrt(delta_num)/2
        return x1,x2
    
    elif len(a)>=3:
        number=0
        while True and number <= num:
            # 正序b
            b=[0]*len(a)
            n=len(a)-1
            # 倒叙
            for i in range(len(a)):
                if i==0:
                    b[0]=a[0]
                elif i==1:
                    b[1]=a[1] - u*b[0]
                else:
                    b[i]=a[i] - u*b[i-1] - v*b[i-2]
            # 此时a、b同长,r0 r1
            r0=b[n-1]
            r1=b[n] + u*b[n-1]
            
            # 计算C
            c=[0]*len(a)
            for i in range(len(a)):
                if i==0:
                    c[0]=b[0]
                elif i==1:
                    c[1]=b[1] - u*b[0]
                else:
                    c[i]=b[i] - u*c[i-1] -v*c[i-2]
            
            # c、b、a同长，从n位向下
            s0=c[n-3]
            s1=c[n-2] + u*c[n-3]
            
            r0_v=-s0
            r1_v=-s1
            r0_u=u*s0-s1
            r1_u=v*s0
            
            mat1=np.mat([[r0_u,r0_v],[r1_u,r1_v]])
            mat2=np.mat([-r0,-r1]).T
    #        mat2=np.mat([r0,r1]).T
            delta=np.linalg.solve(mat1,mat2)
            du = np.array(delta)[0][0]
            dv = np.array(delta)[1][0]
            u += du
            v += dv
        
            if max(abs(du),abs(dv)) < theta:
                break
            
            number += 1
            
#            if (number+1)%500==0:
#                print("正在收敛：",u,":",v)
        delta_num = (u**2-4*v)
        x1=0
        x2=0
        if (delta_num < 0):
            x1=complex(-u/2,np.sqrt(abs(delta_num))/2)
            x2=complex(-u/2,-np.sqrt(abs(delta_num))/2)
        else:
            x1 = -u/2+np.sqrt(delta_num)/2
            x2 = -u/2-np.sqrt(delta_num)/2
            

        return x1, x2, b[:-2]

xl=Bairstow(coefficient,input_u,iuput_v,num = num, theta =theta)
while len(xl)>=2: 
    root_list.append(xl[0])
    root_list.append(xl[1])
    xl=Bairstow(xl[2],xl[0],xl[1],num = num, theta =theta)
if xl[0] != None:
    root_list.append(xl[0])
    
# 打印根
print('多项式根的解 X =\n ',[round(i) for i in root_list])

# 设置图像x**5-7*x**4-3*x**3+79*x**2-46*x-120
fig_1 = plt.figure(figsize = (8, 6))
plt.hlines(0,-5,6,'black','--')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('$f(x)=x^5-7x^4-3x^3+79x^2-46x-120$ 图像的解')


x = np.arange(-4,6,0.05)
y = f(x)
plt.plot(x,y)
plt.scatter(root_list,[0,0,0,0,0],c='black')
plt.show()

