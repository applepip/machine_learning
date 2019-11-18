'''
CART 训练算法
'''

# Scikit-Learn 用分裂回归树（Classification And Regression Tree，简称 CART）算法训练决
# 策树（也叫“增长树”）。这种算法思想真的非常简单：
# 首先使用单个特征 k 和阈值 （例如，“花瓣长度 ≤2.45cm ”）将训练集分成两个子集。它如
# 何选择 k 和 呢？它寻找到能够产生最纯粹的子集一对 ，然后通过子集大小加权计算。当它成功
# 的将训练集分成两部分之后， 它将会继续使用相同的递归式逻辑继续的分割子集，然后是子集的
# 子集。当达到预定的最大深度之后将会停止分裂（由 max_depth 超参数决定），或者是它找不到
# 可以继续降低不纯度的分裂方法的时候。几个其他超参数（之后介绍）控制了其他的停止生长条件
# （ min_samples_split ， min_samples_leaf ， min_weight_fraction_leaf ， max_leaf_nodes ）。

'''
基尼不纯度或是信息熵
'''

# 通常，算法使用 Gini 不纯度来进行检测， 但是你也可以通过将标准超参数设置为 "entropy" 来使用
# 熵不纯度进行检测。这里熵的概念是源于热力学中分子混乱程度的概念，当分子井然有序的时候，熵值
# 接近于 0。

# 熵这个概念后来逐渐被扩展到了各个领域，其中包括香农的信息理论，这个理论被用于测算一段信息中的
# 平均信息密度。当所有信息相同的时候熵被定义为零。熵经常被用作不纯度的衡量方式，当一个集合内只
# 包含一类实例时， 我们称为数据集的熵为 0。基尼指数会趋于在树的分支中将最多的类隔离出来，而熵
# 指数趋向于产生略微平衡一些的决策树模型。

'''
回归
'''

# 注意每个区域的预测值总是该区域中实例的平均目标值。算法以一种使大
# 多数训练实例尽可能接近该预测值的方式分割每个区域。

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

# CART 算法的工作方式与之前处理分类模型基本一样，不同之处在于，现在不再以最小化不
# 纯度的方式分割训练集，而是试图以最小化 MSE（均方误差） 的方式分割训练集。

