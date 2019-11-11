import pandas as pd

def load_housing_data():
    csv_path = "data/housing.csv"
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.head()) #列出数据的前五行

'''
每一行都表示一个街区。共有 10 个属性：经度、维度、房屋年龄中位数、总房间数、
总卧室数、人口数、家庭数、收入中位数、房屋价值中位数、离大海距离。
'''

print(housing.info())  #查看数据的描述，特别是总行数、每个属性的类型和非空值的数量

print(housing["ocean_proximity"].value_counts()) #查看对象中数据类别，每个类别中都包含有多少个街区

print(housing.describe()) #数据属性展示 std：标准差，25%、50%、75% 展示了
                          #对应的分位数：每个分位数指明小于这个值，且指定分组的百分比。

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()