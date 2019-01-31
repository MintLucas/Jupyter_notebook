#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 15:49:48 2017

@author: lamahamadeh
"""

'''
Title: Pima Indians Diabetes Database

2. Sources:
   (a) Original owners: National Institute of Diabetes and Digestive and
                        Kidney Diseases
   (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
                          Research Center, RMI Group Leader
                          Applied Physics Laboratory
                          The Johns Hopkins University
                          Johns Hopkins Road
                          Laurel, MD 20707
                          (301) 953-6231
   (c) Date received: 9 May 1990
   
'''

#important necessary libraries
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#-------------------

#loading the dataframe
df = pd.read_csv('/Users/lamahamadeh/Desktop/pima_indians_diabetes/pima_indians_diabetes.txt')

#-------------------

#defining the columns 
df.columns =['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 
             'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class']
#-------------------

#checking the dataframe
print(df.head())
print(df.dtypes)
print(df.shape) #(767, 9)

#identify nans
def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print ("Missing values per column:")
print (df.apply(num_missing, axis=0),'\n') #no nans

#-------------------

#Apply the K nearest neighbour classifier

#split the data into training and testing datasets
X = np.array(df.drop(['Class'], axis = 1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size =0.5, 
                                                    random_state = 7)

#apply the knn method
Knn = KNeighborsClassifier(n_neighbors = 2)

#train the data
Knn.fit(X_train,y_train)

#test the data
accuracy = Knn.score(X_test, y_test)#this to see how accurate the algorithm is in terms 
#of defining the diabetes to be either 1 or 0
print('accuracy of the model is: ', accuracy) #0.73

#-------------------

#Plotting and visualisation (focus on only two features from the dataset)

X1 = np.array(df[['Plasma_glucose','Age']]) #choose only two features
Y = np.array(df['Class']) #the label of the dataset

h = .02  # step size in the mesh
 
# Create color maps using hex_colors
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])


# apply Neighbours Classifier and fit the data.
X_train, X_test, Y_train, Y_test = train_test_split (X1, y, test_size=0.5, random_state = 7)
Knn = KNeighborsClassifier(n_neighbors = 15)
Knn.fit(X1, y)
 
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = Knn.predict(np.c_[xx.ravel(), yy.ravel()])
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(X1[:, 0], X1[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
plt.ylabel('Age')
plt.title('K = 15')

plt.show()

#-------------------