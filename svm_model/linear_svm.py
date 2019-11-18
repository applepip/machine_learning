'''
支持向量机（SVM）
'''

'''
如果我们严格地规定所有的数据都不在“街道”上，都在正确地两边，称为硬间隔分类，硬间隔
分类有两个问题，第一，只对线性可分的数据起作用，第二，对异常点敏感。
软间隔分类:
在 Scikit-Learn 库的 SVM 类，你可以用 C 超参数（惩罚系数）来控制这种平衡：较小
的 C 会导致更宽的“街道”，但更多的间隔违规。
'''

# 以下的 Scikit-Learn 代码加载了内置的鸢尾花（Iris）数据集，缩放特征，并训练一个线性
# SVM 模型（使用 LinearSVC 类，超参数 C=1 ，hinge 损失函数）来检测 Virginica 鸢尾花,
# 不同于 Logistic 回归分类器，SVM 分类器不会输出每个类别的概率

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ))
svm_clf.fit(X, y)