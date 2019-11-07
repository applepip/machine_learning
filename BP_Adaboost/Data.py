import scipy.io as scio
import numpy as np

data_path="data/data.mat"
data = scio.loadmat(data_path)

input_train = np.array(data['input_train']).T
output_train = np.array(data['output_train']).T
input_test = np.array(data['input_test']).T
output_test = np.array(data['output_test']).T

labels = ['成分费用利润率', '资产营运能力', '公司总资产', '总资产增长率',
          '流动比率', '营业现金流量', '审计意见类型', '每股收益',
          '每股收益', '资产负债率']




