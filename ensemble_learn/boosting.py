'''
提升
'''

# 提升（Boosting，最初称为假设增强）指的是可以将几个弱学习者组合成强学习者的集成方
# 法。对于大多数的提升方法的思想就是按顺序去训练分类器，每一个都要尝试修正前面的分
# 类。现如今已经有很多的提升方法了，但最著名的就是 Adaboost（适应性提升，是 Adaptive
# Boosting 的简称） 和 Gradient Boosting（梯度提升）。

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

'''
Adaboost

'''

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,
algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)

'''
梯度提升
'''

# 让我们通过一个使用决策树当做基分类器的简单的回归例子（回归当然也可以使用梯度提
# 升）。这被叫做梯度提升回归树（GBRT，Gradient Tree Boosting 或者 Gradient Boosted
# Regression Trees）。首先我们用 DecisionTreeRegressor 去拟合训练集（例如一个有噪二次
# 训练集）：

import numpy as np

np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

from sklearn.tree import DecisionTreeRegressor
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

# 现在在第一个分类器的残差上训练第二个分类器：
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

# 随后在第二个分类器的残差上训练第三个分类器：

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)

# 现在我们有了一个包含三个回归器的集成。它可以通过集成所有树的预测来在一个新的实例
# 上进行预测。

X_new = np.array([[0.8]])

y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

print(y_pred)

# 可以使用 sklean 中的 GradientBoostingRegressor 来训练 GBRT 集成。
# 超参数 learning_rate 确立了每个树的贡献。

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
gbrt_y_pred=gbrt.predict(X_new)
print(gbrt_y_pred)

# Gradient Boosting with Early stopping

# 为了找到树的最优数量，你可以使用早停技术。最简单使用这个技术的方法就是使用 staged_predict() ：它
# 在训练的每个阶段（用一棵树，两棵树等）返回一个迭代器。

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

min_error = np.min(errors)

import matplotlib.pyplot as plt


def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.show()

