# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 19:39:18 2018

@author: 14561
"""

import pandas as pd
import numpy as np
#划分训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


epoch=200
batch_size=100




def transform_labels(labels):
    labels=labels.tolist()
    for i in range(len(labels)):
        if(labels[i]==' >50K'):
            labels[i]=1
        else:
            labels[i]=-1
    return np.array(labels)

#data=pd.read_csv('C:/Users/14561/Desktop/adult.csv')
#data=pd.read_csv("C:\\Users\\14561\\Desktop\\adult.csv")
data=pd.read_csv('adult.csv')

'''
数据的类别信息描述：
age:连续性数值变量；
workcass:雇主类型，多类别变量；
fnlwgt:人口普查员认为观察值的人数；
education:教育程度；
education_num:受教育年限；
marital-status:婚姻状况
occupation:职业
relationship:群体性关系
race:种族；
sex:性别;
captical-gain:资本收益;
captical-loss:资本损失;
hours-per-week:每周工作时间
native-country:国籍
result:结果，
'''
#加title，消除空指数据
col_names=["age","workclass","fnlwgt","education","education_num","marital-status","occupation","relationship","race","sex","captal-gain","captical-loss","hours-per-week","native-country","result"]
data=pd.read_csv('adult.csv',names=col_names)
data_clean=data.replace(regex=[r'\?|\.|\$'],value=np.nan)
adult=data_clean.dropna(how='any')
#剔除没有用的数据特征
adult=adult.drop(['fnlwgt'],axis=1)
col_final_names=["age","workclass","education","education_num","marital-status","occupation","relationship","race","sex","captal-gain","captical-loss","hours-per-week","native-country","result"]
x_train,x_test,y_train,y_test=train_test_split(adult[col_final_names[0:13]],adult[col_final_names[13]],test_size=0.25,random_state=33)

#3特征处理
dict_vect=DictVectorizer(sparse=False)#不产生系数矩阵
x_final_train=dict_vect.fit_transform(x_train.to_dict(orient='record'))
#对训练集先用fit_transform然后对测试集使用transform，目的是将两类数据集按照同一样的标准进行处理
x_final_test=dict_vect.transform(x_test.to_dict(orient='record'))

#归一化处理
sc=StandardScaler()
sc.fit(x_final_train)
x_train_std=sc.transform(x_final_train)
x_test_std=sc.transform(x_final_test)

#标签处理
train_labels=transform_labels(y_train)
test_labels=transform_labels(y_test)


#调用库函数
ppn=Perceptron(max_iter=1000,eta0=0.1,random_state=1)
ppn.fit(x_train_std,train_labels)
y_pred=ppn.predict(x_test_std)
miss_classified=(y_pred!=test_labels).sum()
print("sklearn的Perceptron模块的感知机->Accuracy:",(len(test_labels)-miss_classified)/len(test_labels))


#4构建感知机模型
class myPerception:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        
    def dot(self,xi,w,b):
        yPre=np.dot(w,xi)+b
        return int(yPre)
    
    def accuracy(self,w,b,labels):
        count=0
        result=np.dot(self.x,w)+b
        for i in range(len(result)):
            if result[i]*labels[i]>0:
                count+=1
        return count*1.0/len(result)
    
    def predication(self):
        length=np.shape(self.x)[0]
        yita=0.01
        w_bat=np.zeros([self.x.shape[1],1])
        w=np.zeros([self.x.shape[1],1])
        b_bat=0
        b=0
        flag=True
        epoch=0
        batch_size=10000
        Accuracy=0
        while flag:
            epoch+=1
            for j in range(int(length/batch_size)):
                for i in range(batch_size):
                    tempy=self.dot(self.x[i,:],w.T,b)
                    if(self.y[i]*tempy<=0):
                        tmp=yita*self.y[i]*self.x[i]
                        tmp=tmp.reshape(w.shape)
                        w_bat+=tmp
                        b_bat+=yita*self.y[i]
                w+=(w_bat/batch_size)
                b+=(b_bat/batch_size)
            Accuracy=self.accuracy(w,b,self.y)
            if(Accuracy>0.75 and epoch >50):
                flag=False
                print('Epoch:',epoch,'Training_Accuracy:',Accuracy)
                return w,b
            print('Epoch:',epoch,'Training_Accuracy:',Accuracy)
    
def predict(test_data,w,b,test_labels):
    label=[]
    result=0
    count=0
    for i in range(len(test_data)):
        result=np.dot(test_data[i],w)+b
        if(result*test_labels[i]>0):
            result=1
            count+=1
            label.append(result)
        else:
            result=0
            label.append(result)
    return label,count*1.0/len(test_data)


if __name__=='__main__':
    mp=myPerception(x_train_std,train_labels)
    w,b=mp.predication()
    label,test_Accuracy=predict(x_test_std,w,b,test_labels)
    print('Test_Accuracy:',test_Accuracy)
    
            
            






