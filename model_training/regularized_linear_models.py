'''
线性模型的正则化
'''
# 降低模型的过拟合的好方法是正则化这个模型，正则化强制模型有一个小的斜率，
# 它对训练数据的拟合不是那么好，但是对新样本的推广效果好

'''
Ridge回归:（也称为 Tikhonov 正则化）是线性回归的正则化版：在损失函数上直
接加上一个正则项。这使得学习算法不仅能够拟合数据，而且能够使模型的参数权
重尽量的小。这个正则项只有在训练过程中才会被加到损失函数。当得到完成训练
的模型后，我们应该使用没有正则化的测量方法去评价模型的表现。

一般情况下，训练过程使用的损失函数和测试过程使用的评价函数是不一样的。除了正
则化，还有一个不同：训练时的损失函数应该在优化过程中易于求导，而在测试过程
中，评价函数更应该接近最后的客观表现。一个好的例子：在分类训练中我们使用对数
损失（马上我们会讨论它）作为损失函数，但是我们却使用精确率/召回率来作为它的评
价函数。

正则项的超参数决定了你想正则化这个模型的强度。如果正则项的超参数那此时的Ridge回归
便变为了线性回归。如果正则项的超参数非常的大，所有的权重最后都接近于零，最后结果
将是一条穿过数据平均值的水平直线。

提示：
在使用Ridge回归前，对数据进行放缩（可以使用 StandardScaler ）是非常重要的，算法对
于输入特征的数值尺度（scale）非常敏感，大多数的正则化模型都是这样的。
'''

from sklearn.linear_model import Ridge
import numpy as np

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8,4))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)

plt.show()

'''
Lasso 回归：Lasso 回归（也称 Least Absolute Shrinkage，或者 Selection Operator Regression）是另一
种正则化版的线性回归：就像Ridge回那样，它也在损失函数上添加了一个正则化项，Lasso 回归的一个重要特
征是它倾向于完全消除最不重要的特征的权重（即将它们设置为零），即Lasso回归自动的进行特征选择同时输
出一个稀疏模型。
'''

'''
弹性网络（ElasticNet）:弹性网络介于 Ridge 回归和 Lasso 回归之间。它的正则项是 Ridge 回归和 Lasso 
回归正则项的简单混合，同时你可以控制它们的混合率 ，当 时，弹性网络就是 Ridge 回归，当时，其就是 
Lasso 回归。
'''

'''
早期停止法（Early Stopping）
对于迭代学习算法，有一种非常特殊的正则化方法，就像梯度下降在验证错误达到最小值时
立即停止训练那样。我们称为早期停止法。例如使用批量梯度下降来训练一个非常复杂的模
型（一个高阶多项式回归模型）。随着训练的进行，算法一直学习，它在训练集上的预测误
差（RMSE）自然而然的下降。然而一段时间后，验证误差停止下降，并开始上升。这意味着
模型在训练集上开始出现过拟合。一旦验证错误达到最小值，便提早停止训练。这种简单有
效的正则化方法被 Geoffrey Hinton 称为“完美的免费午餐”。
'''

minimum_val_error = float("inf")

'''
Logistic回归:一些回归算法也可以用于分类（反之亦然），Logistic 回归模型计算输入特征
的加权和（加上偏差项），但它不像线性回归模型那样直接输出结果，而是把结果输入 logistic()
函数进行二次加工后进行输出，它通过估计概率并进行预测。
'''