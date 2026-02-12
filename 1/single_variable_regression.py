#!/usr/bin/env python
# coding: utf-8

# # 单变量线性回归
# ## 导入模块

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## 读取训练样本

# In[2]:


def readSample(path):
    data = pd.read_csv(path, header=None, names=['population', 'profit'])
    print(data.head())
    return data


# In[3]:


if __name__ == "__main__":
    path = "D:/Users/l30072207/Documents/Learn/AI/ml/1/ex1data1.txt"
    data = readSample(path)
    data.plot(kind='scatter', x='population', y='profit', figsize=(12,8))


# ## 梯度下降
# ### 代价函数
# $$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$$
# $$h_{\theta}(x^{(i)})=\theta_0+\theta_1x^{(i)}$$

# In[4]:


def computeCost(X, y, theta):
    inner = np.power(((X @ theta) - y), 2) # ndarray使用@表示矩阵乘法，*表示逐元素相乘
    return np.sum(inner) / (2 * len(X))


# ### 初始化X、y、$\theta$
# 特征X添加$\theta_0$对应特征列

# In[5]:


if __name__ == "__main__":
    data.insert(0, 'ones', 1)
    cols=data.shape[1]
    X = data.iloc[:, :-1]
    y = data.iloc[:, cols-1:cols] # 第二维直接用-1会忽略第一行名称
    print(X.head())
    print(y.head())


# 转换为ndarray矩阵

# In[6]:


if __name__ == "__main__":
    X = X.to_numpy()
    y = y.to_numpy()
    theta = np.zeros((2, 1), dtype=np.float32)
    print(f"X:维度{X.shape}\n{X[:5, :]}")
    print(f"y:维度{y.shape}\n{y[:5]}")
    print(f"theta:维度{theta.shape}\n{theta}")


# ### 计算初始代价函数值

# In[7]:


if __name__ == "__main__":
    cost = computeCost(X, y, theta)
    print(f"代价函数初始值为：{cost:.2f}")


# ### 梯度下降算法求解使代价函数局部最小时的$\theta$
# $$\begin{align*}
# \theta_j&=\theta_j-\alpha\frac{\partial}{\partial \theta_j}J(\theta)\\
# &=\theta_j-\frac{\alpha}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}\quad(j=0,1,...,n)
# \end{align*}$$
# 注意：需同时更新$\theta_j$

# In[9]:


def gradientDescent(X, y, theta, alpha, iters):
    cost = np.zeros(iters+1)

    for iter in range(iters):
        cost[iter] = computeCost(X, y, theta)
        error = X @ theta - y
        grad = 1 / len(X) * (X.T @ error)
        theta -= alpha * grad
    cost[iter+1] = computeCost(X, y, theta)
    return theta, cost

def plotFitCurve(data, theta):
    x = np.linspace(data.population.min(), data.profit.max(), 100)
    h = theta[0,0] + theta[1,0] * x

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, h, 'r', label='Prediction')
    ax.scatter(data.population, data.profit, label='Training Data')
    ax.legend() # 自动显示图例
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')

def plotCostConvergence(cost, alpha):
    plt.figure(figsize=(12,8))

    # 绘制曲线
    plt.plot(range(len(cost)), cost, 'r', linewidth=2, label=f"alpha={alpha}")

    # 添加装饰 (f-string 用法)
    plt.title(f"Cost Function Convergence (Final Cost: {cost[-1]:.4f})")
    plt.xlabel("Iterations")
    plt.ylabel("Cost J(theta)")

    # 添加网格方便观察
    plt.grid(True, linestyle='--', alpha=0.6)

def plot_3d(W, B, cost_plot):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(W, B, cost_plot, cmap='rainbow')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel('cost')
    ax.set_title('3D Surface Plot of Cost Function')
    plt.show()

def plot_contour_levels(W, B, cost_plot):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(122)
    contour = ax.contour(W, B, cost_plot, cmap='rainbow', levels=20)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_title('Contour Plot of Cost Funtion with Levels')
    ax.scatter(-3.8, 1.0, color='red', marker='x', s=100, label='Minimum')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    alpha = 0.01
    iters = 1500
    theta, cost = gradientDescent(X, y, theta, alpha, iters)

    plotCostConvergence(cost, alpha)

    print(f"predict1:{np.array([[1,3.5]])@theta}") # 预期输出predict1:[[0.45197648]]
    print(f"predict2:{np.array([[1,7]])@theta}") # 预期输出predict2:[[4.53424489]]

    plotFitCurve(data, theta)

    # 3D可视化代价函数
    theta_0 = np.array(np.linspace(-20, 20, 100)).reshape(-1, 1)
    theta_1 = np.array(np.linspace(-10, 10, 100)).reshape(-1, 1)
    W, B = np.meshgrid(theta_0, theta_1)
    cost_plot = np.zeros_like(W)
    for i in range(len(theta_0)):
        for j in range(len(theta_1)):
            cost_plot[i, j] = computeCost(X, y, np.array([W[i, j], B[i, j]]).reshape(-1, 1))
    plot_3d(W, B, cost_plot)
    plot_contour_levels(W, B, cost_plot)