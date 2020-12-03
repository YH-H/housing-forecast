# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:17:47 2020

@author: yhh
"""

from __future__ import print_function
# 导入相关python库
import numpy as np
import pandas as pd
#设定随机数种子
np.random.seed(36)
import matplotlib.pyplot as plot
from sklearn import metrics
import matplotlib.pyplot as plt

#读取数据
housing = pd.read_csv('C:/Users/Desktop/train_2.csv')
train_data = pd.read_csv('C:/Users/Desktop/1.house-prices-advanced-regression-techniques/train.csv', index_col=0)
col_1 = train_data["SalePrice"]  
target = np.array(col_1)
print("train_SalePrice",target)
#target=pd.read_csv('C:/Users/Desktop/train_2.csv')  #销售价格
t=pd.read_csv('C:/Users/Desktop/test_2.csv')         #测试数据

#数据预处理
housing.info()    #查看是否有缺失值

#特征缩放
from sklearn.preprocessing import MinMaxScaler
minmax_scaler=MinMaxScaler()
minmax_scaler.fit(housing)   #进行内部拟合，内部参数会发生变化
scaler_housing=minmax_scaler.transform(housing)
scaler_housing=pd.DataFrame(scaler_housing,columns=housing.columns)

mm=MinMaxScaler()
mm.fit(t)
scaler_t=mm.transform(t)
scaler_t=pd.DataFrame(scaler_t,columns=t.columns)



#选择基于梯度下降的线性回归模型
from sklearn.linear_model import LinearRegression
LR_reg=LinearRegression()
#进行拟合
LR_reg.fit(scaler_housing,target)



#使用均方误差用于评价模型好坏
from sklearn.metrics import mean_squared_error
preds=LR_reg.predict(scaler_housing)   #输入数据进行预测得到结果
mse=mean_squared_error(preds,target)   #使用均方误差来评价模型好坏，可以输出mse进行查看评价值

#绘图进行比较
plot.figure(figsize=(10,7))       #画布大小
num=100
x=np.arange(1,num+1)              #取100个点进行比较
plot.plot(x,target[:num],label='target')      #目标取值
plot.plot(x,preds[:num],label='preds')        #预测取值
plot.legend(loc='upper right')  #线条显示位置
plot.show()




#输出测试数据
result=LR_reg.predict(scaler_t)
df_result=pd.DataFrame(result)
df_result.to_csv("result.csv")

#输出模型参数、评价模型
print(LR_reg.coef_)
print(LR_reg.intercept_)
#mse_test=np.sum((target-preds)**2)/len(preds) #跟数学公式一样的
#print(mse_test)
print("MSE:",metrics.mean_squared_error(target,preds))
print("RMSE:",np.sqrt(metrics.mean_squared_error(target,preds)))



def try_different_method(clf):
    clf.fit(housing,target)
#    score = clf.score(t, preds)
    result = clf.predict(t)
    
    plt.figure()
    plt.plot(np.arange(len(result)), target,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    #plt.title('score: %f'%score)
    plt.legend()
    plt.show()

#SVM
from sklearn.svm import SVR
#from sklearn import svm
svr = SVR(C=1.0,kernel='rbf', gamma='auto')
try_different_method(svr)

#knn
from sklearn import neighbors
knn = neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=1)
try_different_method(knn)

#决策树
from sklearn import tree
tree_reg = tree.DecisionTreeRegressor()
try_different_method(tree_reg)

#随机森林
from sklearn import ensemble
rf =ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
try_different_method(rf)


from sklearn.metrics import accuracy_score
from sklearn.model_selection import  cross_val_score
from sklearn.preprocessing import LabelEncoder
#读取数据
housing = pd.read_csv('C:/Users/Desktop/train_3.csv')
from sklearn.model_selection import train_test_split
train_X, test_X,train_Y,test_Y= train_test_split(housing,target,test_size = .2,random_state = 123)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
PCAX = pca.fit_transform(housing.values)
#PCAY = pca.fit_transform(t)
PCAtrain_X, PCAtest_X,PCAtrain_Y,PCAtest_Y= train_test_split(PCAX,target,test_size = .2,random_state = 123)

# #1.调用knn分类 X是前面PCA降维后的数据
from sklearn import neighbors
features=['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF',
        'GarageCars', 'BsmtFullBath']
X = housing[features]
le = LabelEncoder()
le.fit(housing['SalePrice'])
y = le.transform(housing['SalePrice'])

knn = neighbors.KNeighborsClassifier(5, weights='uniform')
knn.fit(train_X,train_Y)
scores11 = cross_val_score(knn,train_X, train_Y, cv=5, scoring='accuracy')
print("原始数据：")

print("knn分类交叉验证结果：")
print(np.mean(scores11))
result =knn.predict(test_X)
print("knn分类accuracy_score：")
print (accuracy_score(test_Y, result))


#("knn分类评估结果：")
#print ('classification report :')
#print(classification_report(test_Y,predictions11))
#降维
knn.fit(PCAtrain_X,PCAtrain_Y)
scores12 = cross_val_score(knn, PCAtrain_X, PCAtrain_Y, cv=5, scoring='accuracy')
print("降维后：")
print("knn分类交叉验证结果：")
print(np.mean(scores12))
predictions12 =knn.predict(PCAtest_X)
print("knn分类accuracy_score：")
print (accuracy_score(PCAtest_Y, predictions12))
#print("降维后：knn分类评估结果：")
#print ('classification report :')
#print(classification_report(PCAtest_Y,predictions12))


#2.调用决策树分类
from sklearn import tree
le = LabelEncoder()
le.fit(housing['SalePrice'])
y = le.transform(housing['SalePrice'])
dt = tree.DecisionTreeClassifier(max_depth=1,criterion='entropy', splitter='best',presort=True,min_samples_leaf=3)
dt.fit(train_X,train_Y)
scores21 = cross_val_score(dt,train_X,train_Y, cv=5, scoring='accuracy')
print("原始数据：")

print("决策树分类交叉验证结果：")
print(np.mean(scores21))
predictions21 =dt.predict(test_X)
print("决策树分类accuracy_score：")
print (accuracy_score(test_Y, predictions21))
#print("决策树分类评估结果：")
#print ('classification report :')
#print(classification_report(test_Y,predictions21))
#降维
dt.fit(PCAtrain_X,PCAtrain_Y)
scores22 = cross_val_score(dt,PCAtrain_X,PCAtrain_Y, cv=5, scoring='accuracy')
print("降维后：")

print("决策树分类交叉验证结果：")
print(np.mean(scores22))
predictions22 =dt.predict(PCAtest_X)
print("决策树分类accuracy_score：")
print (accuracy_score(PCAtest_Y, predictions22))
#print("降维后：决策树分类评估结果：")
#print ('classification report :')
#print(classification_report(PCAtest_Y,predictions22))


#3.随机森林分类
from sklearn import ensemble
le = LabelEncoder()
le.fit(housing['SalePrice'])
y = le.transform(housing['SalePrice'])
#森林中树木的数量,即基评估器的数量
rf = ensemble.RandomForestClassifier(n_estimators=2)
rf.fit(train_X,train_Y)
scores31 = cross_val_score(rf, train_X, train_Y, cv=5, scoring='accuracy')
print("原始数据：")

print("随机森林分类交叉验证结果：")
print(np.mean(scores31))
predictions31 =rf.predict(test_X)
print("随机森林分类accuracy_score：")
print (accuracy_score(test_Y, predictions31))
#print("随机森林分类评估结果：")
#print ('classification report :')
#print(classification_report(test_Y,predictions31))
#降维
rf.fit(PCAtrain_X,PCAtrain_Y)
scores32 = cross_val_score(rf,PCAtrain_X,PCAtrain_Y, cv=5, scoring='accuracy')
print("降维后：")

print("随机森林分类交叉验证结果：")
print(np.mean(scores32))
predictions32 =rf.predict(PCAtest_X)
print("随机森林分类accuracy_score：")
print (accuracy_score(PCAtest_Y, predictions32))
#print("降维后：随机森林分类评估结果：")
#print ('classification report :')
#print(classification_report(PCAtest_Y,predictions32))

#4.支持向量机,速度比较慢,影响运行速度
from sklearn import svm
svm = svm.SVC()
svm.fit(train_X,train_Y)
scores41 = cross_val_score(svm,train_X,train_Y, cv=5, scoring='accuracy')
print("svm分类交叉验证结果：")
print(np.mean(scores41))
predictions41 =svm.predict(test_X)
print("svm分类accuracy_score：")
print (accuracy_score(test_Y, predictions41))
#print("svm分类评估结果：")
#print ('classification report :')
#print(classification_report(test_Y,predictions41))
# # 降维
svm.fit(PCAtrain_X,PCAtrain_Y)
scores42 = cross_val_score(svm,PCAtrain_X,PCAtrain_Y, cv=5, scoring='accuracy')
print("降维后：svm分类交叉验证结果：")
print(np.mean(scores42))
predictions42 =svm.predict(PCAtest_X)
print("降维后：svm分类accuracy_score：")
print (accuracy_score(PCAtest_Y, predictions42))
# print("降维后：svm分类评估结果：")
# print ('classification report :')
# print(classification_report(PCAtest_Y,predictions42))

