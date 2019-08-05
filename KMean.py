# In[1]:
import numpy as np
import matplotlib.pyplot as plt

class KMean():
    def __init(self):
        self.cluster_c = None   # 记录聚类的中心
        self.array_c = None     # 记录每个样本所属的中心

    # 两点距离
    def computeDistance(self, x1, x2):
        return np.sum(pow(x1 - x2, 2))

    # 更新每个样本点所属的聚类中心
    def updateSampleCenter(self, X):
        for i in range(self.n_samples):
            min_dis = 2* ((self.max_range + 1) ** 2) # 将最大距离初始化
            for j in range(self.n_cluster):
                dist = self.computeDistance(X[i], self.cluster_c[j])
                if dist < min_dis:
                    min_dis = dist
                    self.array_c[i] = j

    # 更新每个聚类的中心点坐标
    def updateClusterCenter(self, X):
        delta = 0
        cluster_c_old = self.cluster_c.copy()
        for j in range(len(self.cluster_c)):
            self.cluster_c[j] = np.sum(X[self.array_c == j], axis=0) / np.sum(self.array_c == j)
        
        diff = np.sum(pow(cluster_c_old - self.cluster_c, 2)) / len(self.cluster_c)
        return diff

    # 训练
    def fit(self, X, n_cluster, thresthold=0.01, max_itr = 1000):
        self.thresthold = thresthold
        self.max_itr = max_itr
        self.n_cluster = n_cluster
        self.n_samples = len(X)
        self.max_range = np.max(X) - np.min(X)

        self.cluster_c = []
        rand_idx = np.random.randint(0, len(X), n_cluster)
        for i in rand_idx:
            self.cluster_c.append(X[i])
        self.cluster_c = np.array(self.cluster_c)
        # 原始类别
        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(self.cluster_c[:, 0], self.cluster_c[:, 1], marker='*')
        plt.show()

        self.array_c = np.empty(self.n_samples, dtype=np.int32) #记录每个点属于哪个中心点

        itr = 0
        while(True): 
            itr = itr+1
            self.updateSampleCenter(X)
            plt.scatter(X[:, 0], X[:, 1], c=self.array_c)
            plt.scatter(self.cluster_c[:, 0], self.cluster_c[:, 1], marker='*', c='r')
            plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
            plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
            plt.title('K-Mean 第{0}次循环'.format(itr))
            plt.show()
            diff = self.updateClusterCenter(X)

            if itr >= max_itr or diff < self.thresthold:
                break
        
        return self.cluster_c, self.array_c, diff, itr

# In[2]:
from sklearn import datasets
def main():
    n_samples = 1500 # 样本数目
    n_centers = 5   # 聚类数目
    # 使用sklearn生成聚类样本
    blobs = datasets.make_blobs(n_samples = n_samples, random_state=1, centers=n_centers)

    # 颜色数组
    colors = np.array([x for x in 'bgrcmyk'])

    X, y  = blobs
    km = KMean()
    cluster_c, array_c, diff, itr = km.fit(X, n_centers, thresthold=0.1)
    print('K-Mean 算法计算了{0}个循环'.format(itr))
    plt.scatter(X[:, 0], X[:, 1], c=array_c)
    plt.show()

if __name__ == '__main__':
    main()