'''
多输出分类 : 多输出-多类分类
'''

# 为了说明这点，我们建立一个系统，它可以去除图片当中的噪音。它将一张混有噪音的图片
# 作为输入，期待它输出一张干净的数字图片，用一个像素强度的数组表示，就像 MNIST 图片
# 那样。注意到这个分类器的输出是多标签的（一个像素一个标签）和每个标签可以有多个值
# （像素强度取值范围从 0 到 255）。所以它是一个多输出分类系统的例子。

# 分类与回归之间的界限是模糊的，比如这个例子。按理说，预测一个像素的强度更类似
# 于一个回归任务，而不是一个分类任务。而且，多输出系统不限于分类任务。你甚至可
# 以让你一个系统给每一个样例都输出多个标签，包括类标签和值标签。

from datasets import *
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

some_index = 5500
plt.subplot(121); plot_digit(X_test_mod[some_index])
plt.subplot(122); plot_digit(y_test_mod[some_index])
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
