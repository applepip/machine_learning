'''
模型微调
'''

# 微调的一种方法是手工调整超参数，直到找到一个好的超参数组合
# 以RandomForestRegressor算法为例，搜索RandomForestRegressor 超参数值

from datasets import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# 参数的最佳组合
print(grid_search.best_params_)
# 最佳的估计器
print(grid_search.best_estimator_)

# 评分
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

'''
随机搜索
'''
# RandomizedSearchCV:它不是尝试所有可能的组合，而是通过选择每个超参数的一个随机
# 值的特定数量的随机组合。

'''
集成方法
'''
# 另一种微调系统的方法是将表现最好的模型组合起来。组合（集成）之后的性能通常要比单
# 独的模型要好（就像随机森林要比单独的决策树要好），特别是当单独模型的误差类型不同
# 时。

'''
分析最佳模型和它们的误差
'''
# 通过分析最佳模型，常常可以获得对问题更深的了解。可以指出每个属性对于做出准确预测的相对重要性
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

# 将重要性分数和属性名放到一起：
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances,attributes), reverse=True))