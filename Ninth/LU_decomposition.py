# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 21:33:25 2018
LU分解
@author: hhuaf
"""
import matplotlib.pyplot as plt
import numpy as np

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')
class LU():

    def __init__(self):
        self.X = None#系数矩阵
        self.A = None
        self.y = None
        self.W = None#系数求值
        self.L = None
        self.U = None
        self.rank = None

    def fit(self, A, y):#0-n
        pass
                
    def predict(self, A ,y):
        if type(y) != np.ndarray:
            y = np.array(y)
        if type(A) != np.ndarray:
            A = np.array(A)
        y = y.reshape([-1,1])
        W = np.zeros(A.shape)
        L=np.eye(A.shape[0])
        # 交换顺序1,列主元
        for i in range(A.shape[0]):
            W[i] = A[i] / A[i].max()#行交换
        rank = []
        W = np.c_[W, np.arange(A.shape[0])]
        # 交换顺序2
        for i in range(W.shape[0]):
            argmax = np.argmax(W[i:, i])
            #添加序号
            rank.append(W[argmax + i, -1])
            #交换
            temp = W[i].copy()
            W[i] = W[argmax + i]
            W[argmax + i] = temp

        rank = [int(i) for i in rank]
        self.rank=rank#方便还原原来的顺序
        self.A = A#原来的系数矩阵A
        self.W = A[rank]#排序后的
        self.y = y[rank]#排序后的
        
        # 普通高斯消元开始
        n = A.shape[0]
        W = np.c_[self.W, self.y].astype(np.float64)
        for k in range(n-1):#原行
            
            for i in range(k + 1, n):#减行
                L[i][k] = (W[i,k]/W[k,k])
                W[i] = W[i]-W[k]*(W[i,k]/W[k,k])
                
        self.W = W#增加维度了，U+y，分裂两个矩阵
        self.U = np.mat(self.W[:,:-1])#U矩阵，上三角
        self.L = np.mat(L)#下三角
        self.y = self.W[:,-1].reshape(n,-1)
        
        self.y = np.mat(self.y)
        print("U:\n",self.U)
        print("L:\n",self.L)

        x_ = np.array(self.U.I*self.y).reshape(-1)
        self.X=x_
        
        return self.X
            
if __name__=="__main__":   
    A=[[1,1,1],
       [2,1,3],
       [5,3,4]]
    Y=[6,13,23]
    
    model = LU()
    model.fit(A,Y)
    x=model.predict(A,Y)
    print('x的解为:',x)
    fig = plt.figure(figsize=(8,6))
    plt.xlabel('序号')
    plt.ylabel('Y')
    x_list=list(range(1,len(x)+1))
    plt.title("LU分解方程组结果")
    plt.scatter(['x'+str(i) for i in x_list],x,c='black')
    plt.show()