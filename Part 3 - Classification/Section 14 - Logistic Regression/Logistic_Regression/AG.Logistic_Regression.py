#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:55:06 2018

@author: alexandergoff
"""

# From data preprocessing template:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting logistic regression - the only time a model is introduced
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the test set
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#THE GRAPH ____________________________________________
#Visualise the training set results

from matplotlib.colors import ListedColormap
#local variables (_set) allows us to reuse this later
X_set, y_set = X_train, y_train

#meshgrid creates every point on the grid that is possible 
#We are creating the boundary of the grid and hence adding and minusing one to 
#make it easier to see all our points and essentially have padding to the axis (hence min and max) 
#Step is the resolution of the placement of dots 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

#contourf is function contour it draws a straight line across the points
#plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green'))
#a point is  x1(age) and x2(salary)
#then predict which group it goes into (z)
#ravel() just makes sure one number is returned. If 0 it means not bought, if 1 it means bought
#alpha is the colour alpha value and cmap maps colours to the values of 0 or 1
#ListedColormap helps colour all the plotted points
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

#now we add all our points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


