#!/usr/bin/env python
# coding: utf-8

# # 单变量线性回归
# ## 导入模块

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## 读取训练样本

# In[2]:


def readSample(path):
    data = pd.read_csv(path, header=None, names=['population', 'profit'])
    print(data.head())
    return data


# In[ ]:


path = "C:/Users/l30072207/Documents/ml/1/ex1data1.txt"
data = readSample(path)
data.plot(kind='scatter', x='population', y='profit', figsize=(12,8))


# ## 1.2 计算损失函数

# In[3]:


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# In[ ]:


if __name__ == "__main__":
    data = readSample()
    plt.show()

