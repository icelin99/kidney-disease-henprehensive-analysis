#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:23:56 2021

@author: xlinbing
"""
import joblib
import pandas as pd
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.preprocessing._label import LabelEncoder
from sklearn import datasets, preprocessing, metrics
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble._forest import RandomForestClassifier
import sklearn.linear_model as sk
import matplotlib.pyplot as plt
from numpy.random.mtrand import np
from pandas.core.frame import DataFrame
from sklearn.model_selection._split import KFold, train_test_split

def main():
    allData = pd.read_excel('bluetooth_earphone.xlsx')
    allData = allData.dropna(axis=1, how='all')  # 删除空列
    
    bins = [0,49,99,149,199,299,399,499,999,1999,4000] # 分箱
    allData['priceGroup'] = pd.cut(allData["促销价"],bins, labels=['0-50','50-100','100-150','150-200','200-300','300-400','400-500','500-1000','1000-2000','2000-4000'])
    encoder(pd.concat([allData], axis=0))  # 行合并 axis=0,（也就是列对齐）
    lett = joblib.load('lett.pkl')  # 标题
    leprc = joblib.load('leprc.pkl') # 价格
    leor = joblib.load('leor.pkl') # 发货地
    lemat = joblib.load('lemat.pkl') # 耳机材质
    ohe = joblib.load('ohe.pkl')
    A,A2, B, B2 = train_test_split(allData[["标题","priceGroup","发货地","耳机材质"]], allData["销量"], test_size = 0.1, random_state=0)
 
    X = A.values  # train data
    y = B.values  # train result
    X2 = A2.values  # test data
    y2 = B2.values  # test result
    print(X[:,2].shape, X[:,0].value_counts)
    # print(X)
    # print(X2)
    # print(y)
    # print(X2)
    # print(y2)
    
    # print(y,y2)
    # # 将y转为一维
    # y = LabelEncoder().fit_transform(y.ravel()) 
    # y2 = LabelEncoder().fit_transform(y2.ravel()) 
    # print(y,y2)
    
    # 标签编码
    X[:,0] = lett.transform(X[:,0])
    X[:,1] = leprc.transform(X[:,1])
    X[:,2] = leor.transform(X[:,2])
    X[:,3] = lemat.transform(X[:,3])
    # print(X)
    X2[:,0] = lett.transform(X2[:,0])
    X2[:,1] = leprc.transform(X2[:,1])
    X2[:,2] = leor.transform(X2[:,2])
    X2[:,3] = lemat.transform(X2[:,3])
    
    # print(X)
    
    # 对离散特征进行独热编码，扩维
    X = np.hstack((X[:, [0]], ohe.transform(X[:, [1, 2, 3]])))
    X2 = np.hstack((X2[:, [0]], ohe.transform(X2[:, [1, 2, 3]])))
    # print(X)

    DecisionTreeClf = DecisionTreeClassifier()
    DecisionTreeClf.fit(X,y)    #############################################
    
    print(X2)
    predTree = DecisionTreeClf.predict(X2)
    y2_scoreTree = DecisionTreeClf.predict_proba(X2)
    print(y2_scoreTree)
    
    mlp = MLPClassifier()
    mlp.fit(X, y)
    
    predMlp = mlp.predict(X2)

    y2_scoreMlp = mlp.predict_proba(X2)[:, 1] 
    
    # lr = sk.LogisticRegressionCV()
    # lr.fit(X, y) #################################

    # predLogic = lr.predict(X2)

    # y2_scoreLogic = lr.predict_proba(X2)[:, 1] 

def encoder(data):
    X = data.loc[:,["标题","priceGroup","发货地","耳机材质"]].values  # 按标签取数据
    Y = data.loc[:,["销量"]].values

    # 类别数据,需要进行标签编码
    # 标题
    lett = preprocessing.LabelEncoder()
    lett = lett.fit(X[:, 0])
    X[:,0] = lett.transform(X[:,0])
    #print(X[:,0])
    # 价格
    leprc = preprocessing.LabelEncoder()
    leprc = leprc.fit(X[:, 1])
    X[:,1] = leprc.transform(X[:,1])
    # 发货地
    leor = preprocessing.LabelEncoder()
    leor = leor.fit(X[:,2])
    X[:,2] = leor.transform(X[:,2])
    # 耳机材质
    lemat = preprocessing.LabelEncoder()
    lemat = lemat.fit(X[:,3])
    X[:,3] = lemat.transform(X[:,3])
    # 将y转为一维
    Y = LabelEncoder().fit_transform(Y.ravel()) 

    # 独热编码 让变量之间的距离相等
    ohe = OneHotEncoder(sparse=False, categories='auto') # handle_unknown='ignore'
    ohe = ohe.fit(X[:, [1,2,3]])
    ohe.transform(X[:, [1,2,3]])
    
    
    # 保存变量
    joblib.dump(lett, 'lett.pkl')
    joblib.dump(leprc, 'leprc.pkl')
    joblib.dump(leor, 'leor.pkl')
    joblib.dump(lemat, 'lemat.pkl')
    joblib.dump(ohe, 'ohe.pkl')
    
    return

if  __name__ =='__main__':
    main()