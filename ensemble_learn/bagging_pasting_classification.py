'''
Bagging 和 Pasting
'''

# 就像之前讲到的，可以通过使用不同的训练算法去得到一些不同的分类器。另一种方法就是
# 对每一个分类器都使用相同的训练算法，但是在不同的训练集上去训练它们。有放回采样被
# 称为装袋（Bagging，是 bootstrap aggregating 的缩写）。无放回采样称为粘贴（pasting）。

# Bagging 和 Pasting 都允许在多个分类器上对训练集进行多次采样，但只有Bagging 允许对同
# 一种分类器上对训练集进行进行多次采样。当所有的分类器被训练后，集成可以通过对所有分
# 类器结果的简单聚合来对新的实例进行预测。聚合函数通常对分类是统计模式（例如硬投票分
# 类器）或者对回归是平均。每一个单独的分类器在如果在原始训练集上都是高偏差，但是聚合
# 降低了偏差和方差。通常情况下，集成的结果是有一个相似的偏差，但是对比与在原始训练集上
# 的单一分类器来讲有更小的方差。

# sklearn 为 Bagging 和 Pasting 提供了一个简单的API： BaggingClassifier 类（或者对于回归
# 可以是 BaggingRegressor 。接下来的代码训练了一个 500 个决策树分类器的集成，每一个都
# 是在数据集上有放回采样 100 个训练实例下进行训练（这是 Bagging 的例子，如果你想尝试
# Pasting，就设置 bootstrap=False ）。 n_jobs 参数告诉 sklearn 用于训练和预测所需要 CPU
# 核的数量。（-1 代表着 sklearn 会使用所有空闲核）：

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

'''
Out-of-Bag 评价
'''
# 对于 Bagging 来说，一些实例可能被一些分类器重复采样，但其他的有可能不会被采样。
# BaggingClassifier 默认采样。 BaggingClassifier 默认是有放回的采样 m 个实例
# （ bootstrap=True ），其中 m 是训练集的大小，这意味着平均下来只有63%的训练实例
# 被每个分类器采样，剩下的37%个没有被采样的训练实例就叫做 Out-of-Bag 实例。

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)

# 根据这个 obb 评估， BaggingClassifier 可以再测试集上达到93.1%的准确率，我们测试
# 一下我们在测试集上得到了 93.6% 的准确率，足够接近了。

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))