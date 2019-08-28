#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandomForest
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Import accuracy_score
from sklearn.metrics import accuracy_score
# Import function to split data
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

data = pd.read_csv('data.csv')
data = data.iloc[:,:-1]

x = data.drop('diagnosis', axis = 1)
y = data.loc[:,'diagnosis']

# Split the data into 70% train and 30% test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = SEED, stratify = y)
# Instatiate a BaggingClassifier 'bc'
rf = RandomForestClassifier(n_estimators = 300, random_state = SEED)
#min_sample_leaf = 0.12

#Fit 'bc' to the training set
rf.fit(x_train, y_train)
# Pedict test set lables
y_pred = rf.predict(x_test)

print("training accuracy: {:.3f}". format(rf.score(x_train,y_train)))
print("testing accuracy: {:.3f}". format(rf.score(x_test,y_test)))

#print("Accuracy of Random Forest Classifier: {:.3f}".format(accuracy))

import matplotlib.pyplot as plt

plt.figure()
importances_rf = pd.Series(rf.feature_importances_,index = x.columns)
sorted_importances_rf = importances_rf.sort_values()

sorted_importances_rf.plot(kind = 'barh',
                          color = 'lightgreen')

plt.show()

