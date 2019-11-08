import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

datapath = 'lifesat/'

# Load the data
oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)

    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)

    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

x = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]


# 选择线性模型
lin_reg_model = LinearRegression()
# 训练模型
lin_reg_model.fit(x, y)

# Make a prediction for Cyprus
x_new = [[22587]]  # Cyprus' GDP per capita
y_pred = lin_reg_model.predict(x_new)
print(y_pred) # outputs [[ 5.96242338]]

# Visualize the data
fig = plt.figure()
fig.suptitle('GDP & happyness')
plt.scatter(x, y)
plt.xlabel('GDP per capita')
plt.ylabel('Life satisfaction')
plt.plot(x, lin_reg_model.predict(x), color='red', linewidth=2)

# 选择KNN模型
knn_lin_reg_model = KNeighborsRegressor(n_neighbors=5)
# 训练模型
knn_lin_reg_model.fit(x, y)

plt.plot(x, knn_lin_reg_model.predict(x), color='orange', linewidth=2)

plt.show()