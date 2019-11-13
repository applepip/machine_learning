from datasets import *

'''
训练一个二分类器
'''
# 训练一个二分类器，能够识别两类别，“是 5”和“非 5”。

y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# 使用随机梯度下降分类器 SGD 训练
# SGD一次只处理一条数据，适合在线学习（online learning）
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

some_digit = X[36000]  # 数字5
print(sgd_clf.predict([some_digit]))