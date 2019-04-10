#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 23:05:42 2019

@author: xander999------Classification using Artificial Neural Network...
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:29])
X[:, 0:29] = imputer.transform(X[:, 0:29])

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu', input_dim = 30))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
''' Here we have 2 hidden layer and batch_size=10 and no epoch=100   '''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


''' Here we have 2 hidden layer and batch_size=30 and no epoch=100   '''
from sklearn.metrics import confusion_matrix
c3 = confusion_matrix(y_test, y_pred)

''' Here we have 2 hidden layer and batch_size=10 and no epoch=150   '''
from sklearn.metrics import confusion_matrix
c4 = confusion_matrix(y_test, y_pred)

''' Here we have 2 hidden layer and batch_size=10 and no epoch=200   '''
from sklearn.metrics import confusion_matrix
c5 = confusion_matrix(y_test, y_pred)

''' Here we have 3 hidden layer and batch_size=10 and no epoch=100   '''
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)





