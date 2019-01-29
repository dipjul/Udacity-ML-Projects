# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:46:36 2019

@author: Dipjul
"""

# Import statements 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = pd.read_csv('titanic_data.csv')
# Assign the features to the variable X, and the labels to the variable y. 
ab = data.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked'], axis=1)
X = ab.iloc[:,:].values
y = data.iloc[:,1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])


onehotencoder = OneHotEncoder(categorical_features = [1])


X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier()

# TODO: Fit the model.
model.fit(X_train, y_train)
# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X_test)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y_test, y_pred)

