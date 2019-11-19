'''
主成分分析（PCA）
'''

# 主成分分析（Principal Component Analysis）是目前为止最流行的降维算法。首先它找到接
# 近数据集分布的超平面，然后将所有的数据都投影到这个超平面上。

import numpy as np

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X2D=pca.fit_transform(X)

# 84.2% 的数据集方差位于第一轴，14.6% 的方差位于第二轴。
print(pca.explained_variance_ratio_)