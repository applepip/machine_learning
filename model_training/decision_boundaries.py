'''
决策边界
'''

# 我们使用鸢尾花数据集来分析 Logistic 回归。 这是一个著名的数据集，其中包含 150 朵三种
# 不同的鸢尾花的萼片和花瓣的长度和宽度。这三种鸢尾花为：Setosa，Versicolor，Virginica。

from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

# 训练一个逻辑回归模型
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

# 在 1.6 厘米左右存在一个决策边界，这时两类情况出现的概率都等于 50%：如果花瓣宽度大
# 于 1.6 厘米，则分类器将预测该花是 Virginica，否则预测它不是（即使它有可能错了）。

plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.show()

'''
Softmax 回归:和 Logistic 回归分类器一样，Softmax 回归分类器将估计概率最高（它只是得分最高的类）的
那类作为预测结果。我们的目标是建立一个模型在目标类别上有着较高的概率（因此其他类别的概率较低），
最小化当前模型的损失函数（交叉熵）可以达到这个目标，当模型对目标类得出了一个较低的概率，其会惩罚
这个模型。 交叉熵通常用于衡量待测类别与目标类别的匹配程度。
'''
# 使用 Softmax 回归对三种鸢尾花进行分类。
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
# 预测一个花瓣长为 5 厘米，宽为 2 厘米的鸢尾花时，你可以问你的模型你它是哪一类鸢尾花，它
# 会回答 94.2% 是 Virginica 花（第二类），或者 5.8% 是其他鸢尾花。
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))