'''
多类分类
'''
# 二分类器只能区分两个类，而多类分类器（也被叫做多项式分类器）可以区分多于两个类。
# 一些算法（比如随机森林分类器或者朴素贝叶斯分类器）可以直接处理多类分类问题。其他
# 一些算法（比如 SVM 分类器或者线性分类器）则是严格的二分类器。然后，有许多策略可以
# 让你用二分类器去执行多类分类。


from datasets import *

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

# Scikit-Learn 实际上训练了 10 个二分类器，每个分类器都产到
# 一张图片的决策数值，选择数值最高的那个类。

some_digit = X[36000]  # 数字5
print(sgd_clf.predict([some_digit]))

some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)

print(np.argmax(some_digit_scores))

# 一个分类器被训练好了之后，它会保存目标类别列表到它的属性 classes_ 中去，按照值
# 排序。
print(sgd_clf.classes_)
print(sgd_clf.classes_[5])

'''
误差分析
'''
# 你可以检查混淆矩阵。你需要使用 cross_val_predict() 做出预测，然后调
# 用 confusion_matrix() 函数。
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)

from matplotlib import pyplot as plt

# 这个混淆矩阵看起来相当好，因为大多数的图片在主对角线上。在主对角线上意味着被分类
# 正确。数字 5 对应的格子看起来比其他数字要暗淡许多。这可能是数据集当中数字 5 的图片
# 比较少，又或者是分类器对于数字 5 的表现不如其他数字那么好。

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# 让我们关注仅包含误差数据的图像呈现。首先你需要将混淆矩阵的每一个值除以相应类别的
# 图片的总数目。这样子，你可以比较错误率，而不是绝对的错误数（这对大的类别不公
# 平）。

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# 用 0 来填充对角线，这样子就只保留了被错误分类的数据。
# 分析混淆矩阵通常可以给你提供深刻的见解去改善你的分类器。
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()