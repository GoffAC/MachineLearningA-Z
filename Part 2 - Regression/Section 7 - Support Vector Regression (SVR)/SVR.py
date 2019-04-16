#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 19:19:26 2018

@author: alexandergoff
"""

#https://www.youtube.com/watch?v=Y6RRHw9uN9o - SVR Explained
"""It is essentially taking a group of results and raising the power of the results (ie 2d -> 3d space)
then you can create a easier seperation between different example datasets. 
In short it is fudging the numbers so that there is a larger difference between right and wrong hence making 
their seperation and hence your accuracy better"""

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# SVR doesn't do feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() #make two else it will try to fit it to both at the same time
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the SVR Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result

#transform needs an array; need to put the equivalent scaled 6.5 value; need the result inverse y transformed
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([6.5]))))

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()