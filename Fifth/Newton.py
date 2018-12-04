# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:04:35 2018

@author: hhuaf
"""
import matplotlib.pyplot as plt
import numpy as np

#可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
plt.style.use('ggplot')

class Newton():

    def __init__(self):
        
        self.W = None
        self.X = None
        
    def fit(self, X, y):#0-n-1
        
        if type(X) != np.ndarray:
            X=np.array(X)
        if type(y) != np.ndarray:
            y=np.array(y)
        W = np.zeros([len(X),len(X)+1])
        W_list=[]
        W[:,0] = X
        W[:,1] = y
        #针对每列
        W_list.append(W[0,1])
        for j in range(2,W.shape[1]):
            #对每行
            for i in range(j-1,W.shape[0]):
                W[i,j] = (W[i,j-1] - W[i-1,j-1]) / (W[i,0] - W[i-j+1,0])
            W_list.append(W[j-1,j])
        self.W =np.array(W_list)
        self.X = np.concatenate([[1],X[:-1]],axis=0)
        return self
  
    def predict(self, X_pre):
        
        if type(X_pre)!=np.ndarray:
            X_pre = np.array(X_pre)
        def fun(x):
            sum=0
            for i in range(len(self.W)):
                w=self.W[i]
                c=1
                for j in range(i):
                    c*=(x-self.X[j+1])
                sum+=(w*c)
            return sum

        return np.array([fun(i) for i in X_pre])

def generate_data(start,end,sample_num):
    xx=np.arange(start,end,0.01)
    yy=np.log(xx)
    plt.plot(xx.reshape(-1),yy)
    select_num=np.random.choice(np.arange(len(xx)),[sample_num])
    return xx.reshape(-1),yy,[xx[i] for i in select_num],[yy[i] for i in select_num]


m=int(input("请输入样本量：(越多越好)\n"))
fig = plt.figure(figsize = (8, 12))
plt.subplot(2,1,1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('原函数图像')
sca_x,sca_y,fit_x,fit_y=generate_data(1,10,m)
model=Newton()
model.fit(fit_x,fit_y)
W_fit=model.W
y_pre=model.predict(sca_x)
plt.subplot(2,1,2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('牛顿插值图像')
plt.scatter(fit_x,fit_y,c='red')
plt.plot(sca_x,y_pre,c='black')
plt.plot(sca_x,sca_y)
plt.legend(['预测','真实'])
plt.show()
