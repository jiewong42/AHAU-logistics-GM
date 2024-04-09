# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:06:25 2023

@author: jie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel表格数据
data = pd.read_excel('聚类之后.xlsx')

# 提取经度、纬度和权重列
longitudes = data.iloc[:, 1].values
latitudes = data.iloc[:, 2].values
weights = data.iloc[:, 3].values.astype(float)

# 归一化权重
weights /= np.sum(weights)

# 计算重心
center_longitude = np.dot(longitudes, weights)
center_latitude = np.dot(latitudes, weights)

# 生成可视化图表
plt.scatter(longitudes, latitudes, c='b', label='McDonald\'s')
plt.scatter(center_longitude, center_latitude, c='r', label='Centroid')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Centroid of McDonald\'s Restaurants')
plt.legend()
plt.show()

# 输出重心的经纬度信息
print('Centroid Longitude:', center_longitude)
print('Centroid Latitude:', center_latitude)
