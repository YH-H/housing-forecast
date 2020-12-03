# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:11:56 2020

@author: yhh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


train_data = pd.read_csv('C:/Users/Desktop/1.house-prices-advanced-regression-techniques/train.csv', index_col=0)
test_data = pd.read_csv('C:/Users/Desktop/1.house-prices-advanced-regression-techniques/test.csv', index_col=0)

print("查看列名:")
print(train_data.columns)

print("查看数据")
print(train_data.head())

print("查看数据集形状")
print(train_data.shape)

print("查看数据集数据类型")
print(train_data.dtypes)

# 利用pandas的describe()函数给出“SalePrice”数据部分统计量
print("部分数据统计量")
print(train_data['SalePrice'].describe())

# SalePrice 数据分布直方图
sns.distplot(train_data['SalePrice'])

#偏值与峰值的具体数据
print("Skewness: %f" % train_data['SalePrice'].skew())   #skew()样本偏度值
print("Kurtosis: %f" % train_data['SalePrice'].kurt())   #kurt()样本峰度值

#SalePrice与其他四个较为重要的因素之间的关系
#其中两个‘space’变量“TotalBsmtSF”和“GrLivArea”，两个‘building’变量“OverallQual”和“YearBuilt”
# GrLivArea的数值分析
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)); #scatter()绘制散点图，ylim限制y轴坐标范围

# TotalBsmtSF的数值分析
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#OverallQual的分类分析
var = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6)) #plt.subplots()创建一个新的图片f，并返回包含了已生成子图对象的Numpy数组ax
fig = sns.boxplot(x=var, y="SalePrice", data=data)  #data为要输入的数据集
fig.axis(ymin=0, ymax=800000);

#YearBuilt的分类分析
var = 'YearBuilt'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);  #x轴刻度旋转90度

#利用相关系数矩阵，可直观地了解这些因素与销售价格的相关度
#相关系数矩阵
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);  
plt.show()

#热地图，vmax图例中最大值，square设定图片为正方形与否


#销售价格相关系数矩阵
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index 
#nlargast()对corrmat切片，只保留‘SalePrice’列前10个最大值的所有行，cols为这些行的索引
cm = np.corrcoef(train_data[cols].values.T)  
#.T转置
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# SalePrice 与相关因素的散点图
sns.set()
cols = ['SalePrice', 'OverallQuGrLival', 'Area','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], size = 2.5)
plt.show();




"""
k = 10 #number ofvariables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
"""

# SalePrice
#使用log1p, 也就是 log(x+1)，避免了复值的问题  “平滑化”（正态化）
#prices = pd.DataFrame({"price":train_data["SalePrice"], "log(price + 1)":np.log1p(train_data["SalePrice"])})
#prices.hist()

#x/y_train则是SalePrice那一列
#x_train = np.log1p(train_data.pop('SalePrice'))
#y_test = np.log1p(test_data.pop('SalePrice'))

all_data = pd.concat((train_data, test_data), axis=0)
print("all_data_shape",all_data.shape)
#print("train_SalePrice",x_train.head())
#print("train_SalePrice",x_train)
#print("test_SalePrice",test_data.head())


col_1 = train_data["SalePrice"]  
x_train = np.array(col_1)
print("train_SalePrice",x_train)


col_2 = test_data["SalePrice"]  
y_test = np.array(col_2)
print("test_SalePrice",y_test)


#数据缺失
print(all_data['MSSubClass'].dtypes)

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

print(all_data['MSSubClass'].value_counts())

print(pd.get_dummies(all_data['MSSubClass'], prefix='MSSubClass').head())

all_dummy_data = pd.get_dummies(all_data)
print(all_dummy_data.head())


print(all_dummy_data.isnull().sum().sort_values(ascending=False).head(10))


mean_cols = all_dummy_data.mean()
print(mean_cols.head(10))

all_dummy_data = all_dummy_data.fillna(mean_cols)

print(all_dummy_data.isnull().sum().sum())



#建立模型
#把数据集分回 训练/测试集
dummy_train_data = all_dummy_data.loc[train_data.index]
dummy_test_data = all_dummy_data.loc[test_data.index]
print(dummy_train_data.shape, dummy_test_data.shape)
X_train = dummy_train_data.values
X_test = dummy_test_data.values

##


traindata_1 = pd.read_csv('C:/Users/Desktop/train_2.csv')
testdata_1 = pd.read_csv('C:/Users/Desktop/test_2.csv')
print(traindata_1.isnull().any())
print(testdata_1.isnull().any())


model = LinearRegression()
model.fit(traindata_1, x_train)
a = model.intercept_
b = model.coef_
print("最佳拟合线:\n截距", a, "\n回归系数：", b)




