# 首先我们将以一个简单的线性回归模型为例，讨论两种不同的训练方法来得到模型的最优解

'''
1. 直接使用封闭方程进行求根运算，得到模型在当前训练集上的最优参数（即在训练集上
使损失函数达到最小值的模型参数）。
2. 使用迭代优化方法：梯度下降（GD），在训练集上，它可以逐渐调整模型参数以获得最
小的损失函数，最终，参数会收敛到和第一种方法相同的的值。
'''

'''
线性回归训练
'''
# 生成一些近似线性的数据
import numpy as np
from matplotlib import pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

# 使用正规方程(The Normal Equation)来计算模型参数
X_b = np.c_[np.ones((100, 1)), X]
# 使用 Numpy 的线性代数模块（ np.linalg ）中的 inv() 函数来计算矩阵
# 的逆，以及 dot() 方法来计算矩阵的乘法。
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# 生产数据的函数实际上是 y = 4+3x+bias
print(theta_best)

# 使用theta进行预测
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

# 画出这个模型的图像
plt.plot(X_new,y_predict,"r-")
plt.plot(X,y,"b.")
plt.axis([0,2,0,15])
plt.show()

# 使用sklearn.linear_model达到同样效果
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

'''
梯度下降训练：
梯度下降的整体思路是通过的迭代来逐渐调整参数使得损失函数达到最小值
'''
# 在梯度下降中一个重要的参数是步长，超参数学习率的值决定了步长的大小。如果学习率太
# 小，必须经过多次迭代，算法才能收敛，这是非常耗时的。另一个方面如果学习率太大，你
# 将跳过最低点，到达山谷的另一面，可能下一次的值比上一次还要大。这可能使的算法是发
# 散的，函数值变得越来越大，永远不可能找到一个好的答案。最后，并不是所有的损失函数
# 看起来都像一个规则的碗。它们可能是洞，山脊，高原和各种不规则的地形，使它们收敛到
# 最小值非常的困难。

# 训练模型意味着找到一组模型参数，这组参数可以在训练集上使得损失函数最小。这是对于模
# 型参数空间的搜索，模型的参数越多，参数空间的维度越多，找到合适的参数越困难。

# 使用梯度下降的过程中，你需要计算每一个theta下损失函数的梯度。换句话说，你需要计算当
# theta变化一点点时，损失函数改变了多少，这称为偏导数。

# 在这个方程中每一步计算时都包含了整个训练集 ，这也是为什么这个算法称为批量梯度下降：
# 每一次训练过程都使用所有的的训练数据。梯度下降的运算规模和特征的数量成正比。

# 一旦求得了方向是上山的梯度向量，你就可以向着相反的方向去下山，这意味着从theta中减去
# 损失函数的偏导数。学习率和梯度向量的积决定了下山时每一步的大小。

eta = 0.1 # 学习率
n_iterations = 1000
m = 100

theta_gd = np.random.randn(2,1) # 随机初始值
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y)
    theta_gd = theta_gd - eta * gradients

print(theta_gd)

'''
随机梯度下降
'''

# 另一方面，由于它的随机性，与批量梯度下降相比，其呈现出更多的不规律性：它到达最小
# 值不是平缓的下降，损失函数会忽高忽低，只是在大体上呈下降趋势。随着时间的推移，它
# 会非常的靠近最小值，但是它不会停止在一个值上，它会一直在这个值附近摆动。因此，当
# 算法停止的时候，最后的参数还不错，但不是最优值。当损失函数很不规则时，随机梯度下
# 降算法能够跳过局部最小值。因此，随机梯度下降在寻找全局最小值上比批量梯度下降表现
# 要好。

# 虽然随机性可以很好的跳过局部最优值，但同时它却不能达到最小值。解决这个难题的一个
# 办法是逐渐降低学习率。 开始时，走的每一步较大（这有助于快速前进同时跳过局部最小
# 值），然后变得越来越小，从而使算法到达全局最小值。 这个过程被称为模拟退火，因为它
# 类似于熔融金属慢慢冷却的冶金学退火过程。

# 决定每次迭代的学习率的函数称为 learning schedule，使用一个简单的 learning schedule
# 来实现随机梯度下降

n_epochs = 50
t0, t1 = 5, 50 #learning_schedule的超参数
def learning_schedule(t):
    return t0 / (t + t1)

theta_rgd = np.random.randn(2,1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta_rgd) - yi)
        eta = learning_schedule(epoch * m + i)
        theta_rgd = theta_rgd - eta * gradients

print(theta_rgd)

'''
小批量梯度下降
'''
# 在迭代的每一步，批量梯度使用整个训练集，随机梯度时候用仅仅一个实例，在小批量梯度下降中，
# 它则使用一个随机的小型实训练模型例集。它比随机梯度的主要优点在于你可以通过矩阵运算的硬
# 件优化得到一个较好的训练表现，尤其当你使用 GPU 进行运算的时候。但是，另一方面，它有可能陷
# 在局部最小值中（在遇到局部最小值问题的情况下，和我们之前看到的线性回归不一样）。如果你使
# 用了一个较好的 learning schedule ，随机梯度和小批量梯度也可以得到最小值。