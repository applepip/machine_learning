from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='data/')

X, y = mnist["data"], mnist["target"]

# import matplotlib
# import matplotlib.pyplot as plt
# some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 打乱训练集
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]