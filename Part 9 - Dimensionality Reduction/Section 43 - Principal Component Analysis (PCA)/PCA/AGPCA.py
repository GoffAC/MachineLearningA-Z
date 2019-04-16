#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:55:06 2018

@author: alexandergoff

"""
#Principal Component Analysis


# From data preprocessing template:
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values #columns up to but not including 13
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Apply PCA here
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) #use None on your first try
X_train = pca.fit_transform(X_train) #here we use fit_transform as we want the pca to be built for our test data only
X_test = pca.transform(X_test)#as we have created our pca from above, we want to transform this only - so that the results out of the training will be in the same "format" as the test
explained_variance = pca.explained_variance_ratio_ # we can now look at the variance to decide how to visualise it - ie 2dimensions cover 60% variance which is okay to do
#now that we know how many we want to use, we can put that into n_components    


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
X_set = X_train
y_set = y_train

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
             alpha = 0.75, cmap = ListedColormap(()))

#now we add all our points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('PCA')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.legend()
plt.show()


