# -*- coding: utf-8 -*-
"""

@author: Rishabh Jain
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
dataset = pd.read_csv('hazelnut.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 11].values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)

from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
lr = LogisticRegression()
lr.fit(X_train, y_train)

#default solves lbfgs and default multi_class auto
y1= lr.predict(X_test)
print(y1)
lr.score(X_test, y_test)

# 10 fold Cross Validation
from sklearn.model_selection import cross_val_score

#train model with cv of 10 
score = cross_val_score(lr, X, y, cv=10)

#print each cv score (accuracy) and average them
print(score)
print(np.mean(score))
