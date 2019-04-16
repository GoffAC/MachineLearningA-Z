#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:22:12 2018

@author: alexandergoff
"""
# Importing the libraies

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import data set
dataset = pd.read_csv('Data.csv')
#store the data from csv
x = dataset.iloc[:, :-1].values 
# ":" means take all the column's data, ":-1" means except the last one
#create the dependant variable vector
y = dataset.iloc[:,3].values
# "[:, 3]" means take all of the column's data of column number 3 which is the "purchased" column

#MISSING DATA
#to avoid missing data it makes sense to, instead of removing the row of data, average the other inputs from over rows and input that as its value
from sklearn.preprocessing import Imputer
#use this library, and ctrl+i to see its documentation
#using its documentation we can automatically replay NaN with mean across columns (or rows/median/mode)
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
#now we apply this bypicking the object selecting columns that are between the beginning of column 1 and beginning of column 3
imputer = imputer.fit(x[:,1:3])
'''now we have imputer ready with its targets and strategy of what to do with missing data
we need to replace those columns by applying "imputer" and "x" '''
x[:,1:3] = imputer.transform(x[:,1:3])

#CATEGORICAL VARIABLES - changing text/category values that will fuck up equations
#import library
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
#: means all values in column, 0 denotes which field column
x[:,0] = labelencoder_X.fit_transform(x[:,0])
#DUMMY ENCODING - turning a column of variables that need coding into multiple columns where the related point has a value of one to denote it is owned to that one variable
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
#need to transform y so that the yes and no anssers are also numerical values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#SPLIT THE TRAINING SET AND THE DATA SET
from sklearn.cross_validation import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)

#Scaling data - it is necessary as typically machine learning models work off euclidean distance. 
#If one value is huge it will dominate and render the other meaningless. As such we try to normalise our numbers [most librarys do this automatically but check]
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)


#visualise results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

