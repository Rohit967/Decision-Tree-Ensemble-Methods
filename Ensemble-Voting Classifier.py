#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
#
#48 to 56 not needed

# voting classifier 
# Bagging (bootstrap aggregation)

"""
Ensemble method 1: Voting Classifier
"""

# Import accurac_score
from sklearn.metrics import accuracy_score
import pandas as pd
# Import function to split data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
# Import the VotingClassifier meta-model
from sklearn.ensemble import VotingClassifier

# Set seed for reproducibility
SEED = 1

data = pd.read_csv('data.csv')
data = data.iloc[:,:-1]

x = data.drop('diagnosis', axis = 1)
y = data.loc[:,'diagnosis']

# Split the data into 70% train and 30% test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = SEED, stratify = y)

# Instantiate inddividual classifiers

lr = LogisticRegression(random_state = SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state = SEED)

# Define a list called classifier that contains
# The tuples (classifier_name, classifier)
classifiers = [('Logistics Regression',lr),
              ('K Nearest Neighbours',knn),
              ('Classification Tree',dt)]

"""
# Itreate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    # fit clf to the training set
    clf.fit(x_train, y_train)
    # Predict the labels of the test set
    y_pred = clf.predict(x_test)
    # Evaluate the accuracy of clf on the test set
    print ('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
"""
# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators = classifiers)
# Fit 'vc' to the train set
vc.fit(x_train, y_train)
# Predict test set labels
y_pred = vc.predict(x_test)

print("training accuracy: {:.3f}". format(vc.score(x_train,y_train)))
print("testing accuracy: {:.3f}". format(vc.score(x_test,y_test)))

# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {:.3f}',format(
        accuracy_score(y_test,y_pred)))
