# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 22:15:48 2018
高斯消元法（方程组）
@author: hhuaf
"""
import matplotlib.pyplot as plt
import numpy as np

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')

class Gaussian():

    def __init__(self):
        self.X = None#系数矩阵
        self.A = None
        self.y = None
        self.W = None#系数求值

    def fit(self, A, y):#
        self.A = A
        self.y = y
        pass
                
    def predict(self, A ,y):
        if type(y) != np.ndarray:
            y = np.array(y)
        if type(A) !=np.ndarray:
            A = np.array(A)
        y = y.reshape([-1,1])
        W = np.zeros(A.shape)
        # 交换顺序1
        for i in range(A.shape[0]):  
            W[i] = A[i] / A[i].max()
        rank = []
        W = np.c_[W, np.arange(A.shape[0])]
        # 交换顺序2
        for i in range(W.shape[0]):
            argmax = np.argmax(W[i:, i])
            rank.append(W[argmax + i, -1])
            temp = W[i].copy()
            W[i] = W[argmax + i]
            W[argmax + i] = temp

        rank = [int(i) for i in rank]

        self.A = A
        self.W = A[rank]
        self.y = y[rank]
        # 普通高斯消元开始
        n = A.shape[0]
        W = np.c_[self.W, self.y]
        for k in range(n-1):
            for i in range(k + 1, n):
                W[i] = W[i]-W[k]*(W[i,k]/W[k,k])
        self.W = W#增加维度了
        #反向迭代
        x=[0]*n
        for i in range(n)[::-1]:#从下向上
             #上三角
            a_sum=0
            bi = self.W[i,-1]
            for h,j in enumerate(range(i+1,n)):
                a_sum += W[i,j] * x[j]
            x[i]=((bi-a_sum)/W[i,i])
        self.X=x
        return self.X
            
if __name__=="__main__":   
    A=[[1,1,1],
       [2,1,3],
       [5,3,4]]
    Y=[6,13,23]
    
    model = Gaussian()
    model.fit(A,Y)
    x=model.predict(A,Y)
    print(x)
    fig = plt.figure(figsize=(8,6))
    plt.xlabel('序号')
    plt.ylabel('Y')
    x_list=list(range(1,len(x)+1))
    plt.scatter(['x'+str(i) for i in x_list],x,c='black')
    