# -*- coding: utf-8 -*-
"""
迭代法
@author: hhuaf
"""
import numpy as np
import matplotlib.pyplot as plt
# input
'''
x0:初始值
theta:阈值
'''

x0=float(input('输入初始点：(例如5,10,15,20。。。)\n'))
theta=1e-5

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')

# 定义函数，构造数值
init_fun = lambda x: x**2-3*x
tran_fun = lambda x: np.sqrt(3*x)

# 函数图像
fig_1 = plt.figure(figsize = (8, 6))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('$f(x)=x^2-3x$ 图像')

# 函数图像
x=[]
if x0>0:
    x = np.arange(-1,x0,0.05)
    plt.hlines(0,-1,x0,'black','--')
else:
    x = np.arange(x0,10,0.05)
    plt.hlines(0,x0,10,'black','--')
y = init_fun(x)



# 迭代法
def iterative(func = tran_fun, x0 = x0,theta = theta):
    number=0
    xi = x0
    while True and number <= 100:
        
        xi = func(x0)       
        plt.vlines(x0,0,init_fun(x0),'blue','--')
        plt.scatter(x0,init_fun(x0),c='black')
        if abs(xi-x0) < theta:
            return xi,number
        x0 = xi
        number += 1
    
# 迭代法计算求解x0
xi,number = iterative(tran_fun, x0, theta)

print('迭代结果：'+str(xi))
print('迭代次数：'+str(number))

## 函数求解

plt.plot(x,y)
plt.show()


