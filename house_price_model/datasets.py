import pandas as pd

def load_housing_data():
    csv_path = "data/housing.csv"
    return pd.read_csv(csv_path)

import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

housing = load_housing_data()
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "test")



# 如果收入是房价中比较重要的属性，那么可以对收入进行分类

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# print(housing["income_cat"])

'''
数据分割
'''
# Scikit-Learn 提供了一些函数，可以用多种方式将数据集分割成多个子集

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 将预测量和标签分开，drop() 创建了一份数据的备份，而不影响 strat_train_set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

'''
数据清洗
'''
# 去掉确实值对应的特征
housing_f = housing.dropna(subset=["total_bedrooms"])  # 滤除缺失数据   # 选项1
housing.drop("total_bedrooms", axis=1) # 选项2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median,inplace=True)  # 替换缺失数据  # 选项3

'''
处理文本和类别属性
'''
# 文本标签转换为数字
housing_cat = housing["ocean_proximity"]
# 映射表和类别
housing_cat_encoded, housing_categories = housing_cat.factorize()
# 制作映射表的独热编码
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_one_hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

# 以上方法的简单操作：从文本分类到整数分类，再从整数分类到独热向量
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)

'''
特征缩放
'''
# 当输入的数值属性量度不同时，机器学习算法的性能都不会好
# 两种缩放方法
# 1. 线性函数归一化（normalization）:值被转变、重新缩放，直到范围变成 0 到 1。[Scikit-Learn 提供了一个转换器 MinMaxScaler 来实现这个功能]
# 2. 标准化（standardization）:首先减去平均值（所以标准化值的平均值总是 0），然后除以方差，使得到的分布具有单位方差。[Scikit-Learn 提供了一个转换
# 器 StandardScaler 来进行标准化]
'''
转换流水线:数据转换步骤，需要按一定的顺序执行。[Scikit-Learn 提供类 Pipeline]
'''
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

from  fun_customize import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    ('std_scaler', StandardScaler()),
    ])

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)