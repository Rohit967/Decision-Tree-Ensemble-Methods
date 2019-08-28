#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Tree, Ensemble & RandomForest Methods with 
Indian Liver Patient Records DataSet.
"""

# =============================================================================
# # Importing necessary modules
# =============================================================================
import pandas as pd
# Import Decision Tree Classifier from sklearn.tress
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier
# import funtion to RandomForrestClassifier
from sklearn.ensemble import RandomForestClassifier
# Import function to split data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

# Import the VotingClassifier meta-model
from sklearn.ensemble import VotingClassifier

# =============================================================================
# # Set seed for reproducibility
# =============================================================================
SEED = 1

# =============================================================================
# # Loading the est dataset: liv
# =============================================================================
liv = pd.read_csv('indian_liver_patient.csv')
liv = liv.iloc[:,:-1]
liv = pd.DataFrame(liv).fillna(0)

# =============================================================================
# # Create freature and target arrays
# =============================================================================
y = liv.loc[:,'Gender']
x = liv.drop(['Gender'], axis = SEED)

# =============================================================================
# # Split the data into 80% train and 20% test
# =============================================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = SEED, 
                                                    stratify = y)

# =============================================================================
# # Decision Tree
# =============================================================================

# Instantiate a DecisionTreeClassifier 'dt' 
dt = DecisionTreeClassifier(criterion = 'gini', random_state=SEED)

#fit 'dt' to the training set
dt.fit(x_train, y_train)

# Predict test set labels
y_pred = dt.predict(x_test)

confusion_matrix(y_pred,y_test)

# Evaluate training and test acc score.
print("")
print("Decision Tree result :-")
print("Training Accuracy: {}". format(dt.score(x_train,y_train)))
print("Testing Accuracy: {}". format(dt.score(x_test,y_test)))

# Generate plot
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus

# Draw a decision tree
dotfile = StringIO()
tree.export_graphviz(dt, out_file = dotfile,filled= True, impurity= True, 
                     node_ids= True, precision = 2 )

graph = pydotplus.graph_from_dot_data(dotfile.getvalue())

graph.write_png("dtree.png")


# =============================================================================
# # Voting Classifier
# =============================================================================

# Instantiate a LogisticRegression 'lr' 
lr = LogisticRegression(random_state = SEED)
# Instantiate KNeighbors Classifier 'knn'
knn = KNN()
# Instantiate Decision Tree Classifier 'dt'
dt = DecisionTreeClassifier(random_state = SEED)

# Define a list called classifier that contains
# The tuples (classifier_name, classifier)
classifiers = [('Logistics Regression',lr),
              ('K Nearest Neighbours',knn),
              ('Classification Tree',dt)]

# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators = classifiers)
# Fit 'vc' to the train set
vc.fit(x_train, y_train)
# Predict test set labels
y_pred = vc.predict(x_test)

# Evaluate training and test acc score.
print("")
print("Voting Classifier result :-")
print("Training Accuracy: {:.3f}". format(vc.score(x_train,y_train)))
print("Testing Accuracy: {:.3f}". format(vc.score(x_test,y_test)))


# =============================================================================
# # Bagging Classifier
# =============================================================================

# Instantiate dt
dt = DecisionTreeClassifier(random_state=6)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, bootstrap=True, n_estimators=60, 
                       random_state=6)

# Fit bc to the training set
bc.fit(x_train, y_train)

# Predict test set labels
y_pred = bc.predict(x_test)

# Evaluate training and test acc score.
print("")
print("Bagging result :-") 
print("Training Accuracy: {:.3f}". format(bc.score(x_train,y_train)))
print("Testing Accuracy: {:.3f}". format(bc.score(x_test,y_test)))


# =============================================================================
# # Random Forest Classifier
# =============================================================================

# Instatiate a RandomForest 'rf'
rf = RandomForestClassifier(n_estimators = 250, random_state = SEED)

#Fit 'rf' to the training set
rf.fit(x_train, y_train)
# Pedict test set lables
y_pred = rf.predict(x_test)

# Evaluate training and test acc score.
print("")
print("Random Forest result :-")
print("Training Accuracy: {:.3f}". format(rf.score(x_train,y_train)))
print("Testing Accuracy: {:.3f}". format(rf.score(x_test,y_test)))

# Generate plot
import matplotlib.pyplot as plt

plt.figure()
# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_,index = x.columns)
# Sort importances
sorted_importances_rf = importances_rf.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.title('Random Forest Regressor for INDIAN LIVER PATIENTS', color='blue', size=16)
sorted_importances_rf.plot(kind = 'barh',
                          color = 'green')

plt.show()


