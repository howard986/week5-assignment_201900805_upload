# 聚类k-mean算法测试
# In[1]:
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from KMean import KMean
from KMeanPlus import KMeanPlus
n_samples = 1500 # 样本数目
n_centers = 5   # 聚类数目
# 使用sklearn生成聚类样本
blobs = datasets.make_blobs(n_samples = n_samples, random_state=1, centers=n_centers)

# 颜色数组
colors = np.array([x for x in 'bgrcmyk'])

X, y  = blobs
# 原始类别
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()

# In[2]:
km = KMean()
cluster_c, array_c, diff, itr = km.fit(X, n_centers)
print('K-Mean 算法计算了{0}个循环'.format(itr))
plt.scatter(X[:, 0], X[:, 1], c=array_c)
plt.show()

# In[3]:
kmp = KMeanPlus()
cluster_c, array_c, diff, itr = kmp.fit(X, n_centers)
print('K-Mean++ 算法计算了{0}个循环'.format(itr))
plt.scatter(X[:, 0], X[:, 1], c=array_c)
plt.show()