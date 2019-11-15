'''
多项式回归
'''
# 多项式回归: 如果数据实际上比简单的直线更复杂，依然可以使用线性模型来拟合非线性数据。
# 一个简单的方法是对每个特征进行加权后作为新的特征，然后训练一个线性模型在这个扩展的
# 特征集。

import numpy as np
from matplotlib import pyplot as plt

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])

from sklearn.preprocessing import PolynomialFeatures
# 使用 Scikit-Learning 的 PolynomialFeatures 类进行训练数据集的转换，让训练集中每个
# 特征的平方（2 次多项式）作为新特征。
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(X_poly[0])

# X_poly 现在包含原始特征并加上了这个特征的平方 。现在你可以在这个扩展训练集上
# 使用 LinearRegression 模型进行拟合

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

X_new=np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()

# 可以使用交叉验证来估计一个模型的泛化能力,另一种方法是观察学习曲线：画出模型在训练集上
# 的表现，同时画出以训练集规模为自变量的训练集函数

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

        plt.xlabel("Training set size", fontsize=18)
        plt.ylabel("RMSE", rotation=0, fontsize=18)
        plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
        plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")


# 简单线性回归模型的学习曲线
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

# 上面的曲线表现了一个典型的欠拟合模型，两条曲线都到达高原地带并趋于稳定，并且最后
# 两条曲线非常接近，同时误差值非常大。

plt.show()

# 在统计和机器学习领域有个重要的理论：一个模型的泛化误差由三个不同误差的和决
# 定：
# 偏差：泛化误差的这部分误差是由于错误的假设决定的。例如实际是一个二次模型，
# 你却假设了一个线性模型。一个高偏差的模型最容易出现欠拟合。
# 方差：这部分误差是由于模型对训练数据的微小变化较为敏感，一个多自由度的模
# 型更容易有高的方差（例如一个高阶多项式模型），因此会导致模型过拟合。
# 不可约误差：这部分误差是由于数据本身的噪声决定的。降低这部分误差的唯一方
# 法就是进行数据清洗（例如：修复数据源，修复坏的传感器，识别和剔除异常值）。