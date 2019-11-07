import matplotlib as plt
import pandas as pd

gdp_per_capita = pd.read_csv("lifesat/gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a", index_col="Country")
gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)



print(gdp_per_capita[0:5]['Estimates Start After'])