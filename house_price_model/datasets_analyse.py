from datasets import *

import matplotlib.pyplot as plt

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")


# 将 alpha 设为 0.1，可以更容易看出数据点的密度
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# 每个圈的半径表示街区的人口（选项 s ），颜色代表价格（选项 c ）。
# 我们用预先定义的名为 jet 的颜色图（选项 cmap ），它的范围是从蓝色（低价）到红色（高价）。
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

# 因为数据集并不是非常大，你可以很容易地使用 corr() 方法计算出每对属性间的
# 标准相关系数（standard correlation coefficient，也称作皮尔逊相关系数）：
# 相关系数的范围是 -1 到 1。当接近 1 时，意味强正相关；例如，当收入中位数增加时，房价
# 中位数也会增加。当相关系数接近 -1 时，意味强负相关；
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# 另一种检测属性间相关系数的方法是使用 Pandas 的 scatter_matrix 函数，它能画出每个数
# 值属性对每个其它数值属性的图

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# 最有希望用来预测房价中位数的属性是收入中位数
housing.plot(kind="scatter", x="median_income",y="median_house_value", alpha=0.1)

# 属性组合试验
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

plt.show()