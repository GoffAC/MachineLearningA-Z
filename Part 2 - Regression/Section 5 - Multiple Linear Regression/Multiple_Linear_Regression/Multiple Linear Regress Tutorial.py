#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 08:38:59 2018

@author: alexandergoff
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap (deleting the first column)
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Multiple linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the results
y_pred = regressor.predict(X_test)

#Backward elimination to find optimum model
import statsmodels.formula.api as sm
#append adds the column(s) at the end, hence if we write it the other way round the new column will be at the front
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
#create the statistically significant x (with reduced variables)- first we import all columns
X_opt = X[:,[0,1,2,3,4,5]]
#create a new regressor to do Backwards Elimination
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
#we need to find the p value to determine which variables are statistically significant and hence decide to remove them in the assessment
regressor_OLS.summary() 
#looking at the p values in the summary "x2" has the largest p value. This corresponds to column 2 (as constant is column 0, etc)
