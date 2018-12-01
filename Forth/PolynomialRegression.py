# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:09:27 2018
多项式回归
@author: hhuaf
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

'''
每次运行不一样
i为样本号
[xi,yi]
yi=Eaixi

数据集有0-n n+1个
高次项有x^0-n 的

先构造0-2n
'''

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')

class PolynomialRegression():

    def __init__(self):
        
        self.W = None
        self.pol=None
        self.degree=None

    def fit(self, X, y, n):
        if type(X)!=np.ndarray:
            X=np.array(X)
        if len(X.shape) <= 2:
            X=X.reshape([-1,1])
            
        if type(y)!=np.ndarray:
            y=np.array(y)
        if len(y.shape)<=2:
            y=y.reshape([-1,1])
        # 样本数
        assert X.shape[0] == y.shape[0]
        
        self.degree=n
        self.W=self.Mat_transform(X,y)
        return self
  
    def Mat_transform(self,X,y):
        
#        self.pol=PolynomialFeatures(degree=2*self.degree)
#        mat_X=np.ones([len(X),len(X)])
#        X_t=self.pol.fit_transform(X)
#
#        X_t=X_t.sum(axis=0) #0-2n
#        # for X
#        for i in range(len(X)):
#            for j in range(len(X)):
#                mat_X[i,j]=X_t[i+j]
#        mat_X=np.mat(mat_X)
#        self.pol=PolynomialFeatures(degree=self.degree)
##        y_t=self.pol.fit_transform(y)
#        y_t=np.concatenate([y]*len(X),axis=1)
#        x_t=self.pol.fit_transform(X)
#        mat_y=(y_t*x_t).sum(axis=0)
#        mat_X=np.mat(mat_X)
#        mat_y=np.mat(mat_y).reshape([-1,1])
#        return mat_X.I*mat_y
        self.pol=PolynomialFeatures(degree=self.degree)
        X_t=np.mat(self.pol.fit_transform(X))
        XX=X_t.T*X_t
        XY=X_t.T*np.mat(y)
        return XX.I*XY
    def predict(self, X):
        if type(X)!=np.ndarray:
            X=np.array(X)
        if len(X.shape)==1:
            X=X.reshape([-1,1])
        X = self.pol.fit_transform(X)
        X=np.mat(X)
        return np.array(X*self.W)

def generate_data(A,start,end,sample_num):
    pol=PolynomialFeatures(len(A)-1)
    xx=np.arange(start,end,0.01).reshape([-1,1])
    x_t=pol.fit_transform(xx)
    yy=(x_t*np.array(A)).sum(axis=1)
    yy=yy+np.random.randn(len(yy))
#    print(xx.shape,":",yy.shape)
#    print(xx)
    plt.scatter(xx.reshape(-1),yy)
    select_num=np.random.choice(np.arange(len(xx)),[sample_num])
    return xx.reshape(-1),yy,[xx[i] for i in select_num],[yy[i] for i in select_num]

n=int(input("请输入最高次项数：\n"))
m=int(input("请输入样本量：(越多越好)\n"))
A=2*np.random.randn(n+1)

fig = plt.figure(figsize = (8, 10))
plt.subplot(2,1,1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('原函数抖动散点图像')

sca_x,sca_y,fit_x,fit_y=generate_data(A,-2,2,m)

model=PolynomialRegression()

model.fit(fit_x,fit_y,n)

A_fit=model.W
print('原系数',A)
print('拟合系数',np.array(A_fit).reshape(-1))
y_pre=model.predict(sca_x)
plt.subplot(2,1,2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('多项式拟合图像')
plt.scatter(fit_x,fit_y)
plt.plot(sca_x,y_pre)
plt.show()
