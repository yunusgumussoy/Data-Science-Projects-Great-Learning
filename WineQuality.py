# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 01:40:38 2023

@author: Yunus

Data Science Projects @ Great Learning

Prediction on Wine Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# data import
data = pd.read_csv('wine.data.csv', names = ['Quality','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'])

# data check
data.head(10)

# variable types
data.dtypes

# checking missing values
data.isnull().sum()

# Analyzing data
# The transpose() function is used to transpose index and columns.
data.describe().transpose()

# plot
sns.pairplot(data, diag_kind = 'kde', hue = 'Quality')

# Building Model
# selecting variables
x = data.drop(columns = 'Quality')
y = data['Quality']

from sklearn.model_selection import train_test_split

# splitting data into training and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=1)

from sklearn.naive_bayes import GaussianNB

# NB Gaussian function for the model
# fitting model
model = GaussianNB()
model.fit(x_train,y_train)

# Testing the accuracy of model
model.score(x_test,y_test)

y_pred = model.predict(x_test)

metrics.confusion_matrix(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

score = cross_val_score(model,x_train,y_train,cv=10)
print('cross validation score : ',score)
print('Average score : ',np.average(score))
