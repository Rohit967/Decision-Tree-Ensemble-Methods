#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# Import Decision Tree Classifier from sklearn.tress
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Import accuracy_score
from sklearn.metrics import accuracy_score


#read the dataset
data = pd.read_csv('data.csv')
data = data.iloc[:,:-1]

#X = data.loc[:,['radius_mean','concave point_mean']]
x = data.drop('diagnosis',axis=1)
y = data.loc[:,'diagnosis']

SEED = 1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = SEED, stratify = y)

# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6
dt = DecisionTreeClassifier(criterion = 'gini', random_state=SEED)
#max_depth = 8,


#fit dt to the training set
dt.fit(x_train, y_train)

# Predict test set labels
y_pred = dt.predict(x_test)
print(y_pred)

confusion_matrix(y_pred,y_test)

print("training accuracy: {}". format(dt.score(x_train,y_train)))
print("testing accuracy: {}". format(dt.score(x_test,y_test)))

from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus

dotfile = StringIO()
tree.export_graphviz(dt, out_file = dotfile,filled= True, impurity= True, node_ids= True,
                     precision = 2 )

graph = pydotplus.graph_from_dot_data(dotfile.getvalue())

graph.write_png("dtree.png")


