'''
多标签分类: 一个样例输出多个类别
'''
from datasets import *
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)  # 该标签指出这个数字是否为大数字（7，8 或者 9）
y_train_odd = (y_train % 2 == 1)   # 该标签指出这个数字是否是奇数
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
some_digit = X[36000]  # 数字5
# 显示数字 5 不是大数（ False ），同时是一个奇数（ True ）。
print(knn_clf.predict([some_digit]))