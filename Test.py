#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

# This code runs 5 different classifiers and chooses the classifer
# that has the lowest training accuracy.
# Classifiers that will be used are (SVM, Decision Trees,
# Perceptron, K-nearst neighbour, Linear Regression)


# Dependices
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestNeighbors


#Methods used
methods = ['Decision Trees', 'SVM', 'Perceptron', 'K nearest neighbour']


# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']



# Initlizae models
clf_tree    = DecisionTreeClassifier()
clf_svm     = SVC()
clf_percept = Perceptron()
clf_KNN     = NearestNeighbors()

# Train all models
clf_tree    = clf_tree.fit(X,Y)
clf_svm     = clf_svm.fit(X,Y)
clf_percept = clf_percept.fit(X,Y)
clf_KNN     = clf_KNN.fit(X,Y)

#Test models on the same training set to find the training accuracy
# Decision Trees
clf_tree_prediction    = clf_tree.predict(X)
acc_tree = accuracy_score(Y, clf_tree_prediction)*100
print ("Accuracy using Decision Trees:"), acc_tree, "%"

#SVM
clf_svm_prediction     = clf_svm.predict(X)
acc_svm = accuracy_score(Y, clf_svm_prediction)*100
print ("Labels for training set using SVM:'"),acc_svm, "%"

#Perceptron
clf_percept_prediction = clf_percept.predict(X)
acc_per = accuracy_score(Y, clf_percept_prediction)*100
print ("Labels for training set using Perceptron:"),acc_per, "%"

#KNN
distances, indices     = clf_KNN.kneighbors(X)
new_label = indices[:,0]
clf_KNN_prediction = [Y[i][:] for i in new_label ]
acc_knn = accuracy_score(Y, clf_KNN_prediction)*100
print ("Labels for training set using K-nearst neighbour:"),acc_knn, "%"


#All accuracies
acc_all = [acc_tree,acc_svm,acc_per,acc_knn]

#Chosing the best among all
score_bestmethod = np.max(acc_all)
best_method = np.argmax(acc_all)

#Print out best method and score
print (methods[best_method], ("is the best method with accuracy of"), score_bestmethod, "%")