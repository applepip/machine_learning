'''
随机森林
'''

# 随机森林是决策树的一种集成，通常是通过 bagging 方法（有时是pasting 方法）进行训练，通
# 常用 max_samples 设置为训练集的大小。与建立一个 BaggingClassifier 然后把它放入 DecisionTreeClassifier
# 相反，你可以使用更方便的也是对决策树优化够的 RandomForestClassifier （对于回归是 RandomForestRegressor ）。
# 接下来的代码训练了带有 500 个树（每个被限制为 16 叶子结点）的决策森林，使用所有空闲的CPU 核：

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

print(y_pred_rf)

'''
特征重要度
'''
# 如果你观察一个单一决策树，重要的特征会出现在更靠近根部的位置，而不重要的特
# 征会经常出现在靠近叶子的位置。因此我们可以通过计算一个特征在森林的全部树中出现的
# 平均深度来预测特征的重要性。sklearn 在训练后会自动计算每个特征的重要度。你可以通
# 过 feature_importances_ 变量来查看结果。


# 以iris 数据集为例，最重要的特征是花瓣长度（44%）和宽度（42%），而萼片长度和宽度
# 相对比较是不重要的（分别为 11% 和2%）：

from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)