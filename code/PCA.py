# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 01:18:13 2020

@author: yhh
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

#加载数据
#从给定的csv文件加载数据
"""
data = []
with open("C:/Users/Desktop/train_2.csv","r") as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    h = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        data.append(row[1:])

Z = [[float(z) for z in row] for row in data] #将字符串转成float类型
Z = np.array(Z)
"""


data = pd.read_csv('C:/Users/Desktop/train_2.csv')
pca = PCA(n_components=2)
PCAX = pca.fit_transform(data)
print(PCAX)
#PCAX = pca.fit_transform(data.iloc[:,:-1].values)#将数据集的（所有行和除最后一列外所有列）的数值导入X矩阵
plt.scatter(PCAX[:,0],PCAX[:,1],marker='o',s=50,c='red')

