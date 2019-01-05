# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 22:15:14 2019
欧拉法+修恩法
@author: hhuaf
"""

import matplotlib.pyplot as plt
import numpy as np

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')

def compute(coef, x):
        y_deri = 0
        for i,j in enumerate(coef[::-1]):
            y_deri += (x**i)*j
        return y_deri

class Euler():          
    def __init__(self, h=0.1, y0=1,):        #初始化Euler类的方法
        self.deri = None
        self.y0 = None
        self.x0 = None
        self.x_list = None
        self.y_list = None
        self.h = None
        self.n = None
    def euler(self, deri, x0, y0, h, start, end):       #定义Euler算法的过程
        self.x0 = x0
        self.y0 = y0
        self.deri = deri
        self.h = h
        self.n = int((end-start)/h)
        self.x_list = [x0]
        self.y_list = [y0]
        for i in range(self.n):
            x_ = self.x_list[-1]
            y_ = self.y_list[-1]
            y_dere = compute(self.deri,x_)
            x_ += self.h
            y_ += self.h * y_dere
            self.x_list.append(x_)
            self.y_list.append(y_)

        return self.x_list,self.y_list
    
    def up_euler(self, deri, x0, y0, h, start, end):#定义改进Euler算法的过程
        self.x0 = x0
        self.y0 = y0
        self.deri = deri
        self.h = h
        self.n = int((end-start)/h)
        self.x_list = [x0]
        self.y_list = [y0]
        #第一步
        for i in range(self.n):
            x_ = self.x_list[-1]
            y_ = self.y_list[-1]
            y_dere0 = compute(self.deri,x_)
            y_dere1 = compute(self.deri,x_ + self.h)
            y_dere=(y_dere0 + y_dere1)/2
            x_ += self.h
            y_ += self.h * y_dere
            self.x_list.append(x_)
            self.y_list.append(y_)
        return self.x_list,self.y_list


if __name__=="__main__":   
    
    true_func = [-0.5, 4, -10, 8.5, 1]
    deri = [-2, 12, -20, 8.5] #3阶方程
    x0=0
    y0=1
    h=0.5
    x_start =0
    x_end =4
    
    model = Euler()
    x, y1 = model.euler(deri, x0, y0, h, x_start, x_end)
    x, y2 = model.up_euler(deri, x0, y0, h, x_start, x_end)
    fig = plt.figure(figsize=(10,8))
    print('欧拉法结果：',y1)
    print('修恩法结果：',y2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("欧拉法求解微分方程$y=f(x)=0.5x^4+4x^3-10x^2+8.5x=1$结果")
    plt.plot(x,y1,c='blue',label='欧拉法')
    plt.plot(x,y2,c='red',label='修恩法')
    plt.plot(x,[compute(true_func,i) for i in x],c='black',label='原函数')
    plt.legend(['欧拉法','修恩法','原函数'])
    plt.legend()
    plt.show()