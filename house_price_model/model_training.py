from datasets import *
from sklearn.linear_model import LinearRegression


'''
线性回归
'''
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("LinearRegression Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))

# 使用 Scikit-Learn 的 mean_squared_error 函数，用全部训练集来计算下这个回归模型的 RMSE：
from sklearn.metrics import mean_squared_error
housing_predictions_l = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions_l)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

'''
决策树回归
'''
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions_t = tree_reg.predict(housing_prepared)
print("DecisionTreeRegressor Predictions:\t", housing_predictions_t)
print("Labels:\t\t", list(some_labels))
tree_mse = mean_squared_error(housing_labels, housing_predictions_t)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

'''
使用交叉验证做更佳的评估
'''
# 交叉评估： 它随机地将训练集分成十个不同的子集，成为“折”，然后训练评
# 估决策树模型 10 次，每次选一个不用的折来做评估，用其它 9 个来做训练。
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("DecisionTreeRegressor notes")
print("Scores:", tree_rmse_scores)
print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())

# 计算线性回归模型的分数
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

print("LinearRegression notes")
print("Scores:", lin_rmse_scores)
print("Mean:", lin_rmse_scores.mean())
print("Standard deviation:", lin_rmse_scores.std())

'''
随机森林
'''
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions_f = forest_reg.predict(housing_prepared)
print("RandomForestRegressor Predictions:\t", housing_predictions_f)
print("Labels:\t\t", list(some_labels))
forest_mse = mean_squared_error(housing_labels, housing_predictions_f)
forest_mse = np.sqrt(forest_mse)
print(forest_mse)
forest_rmse_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-lin_scores)

print("RandomForestRegressor notes")
print("Scores:", forest_rmse_scores)
print("Mean:", forest_rmse_scores.mean())
print("Standard deviation:", forest_rmse_scores.std())