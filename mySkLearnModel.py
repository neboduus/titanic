#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import tree
import collections
import csv as csv

######################################

# Main Program 

dataset = pd.read_csv('train.csv', header=0)
features = dataset.columns.tolist()
m,n = dataset.shape

test_dataset = pd.read_csv('test.csv', header=0)
m_t,n_t = test_dataset.shape

features_e = []
features_e.append(features.pop(0))		#eliminated PassengerId
features_e.append(features.pop(1))		#eliminated Name
features_e.append(features.pop(5))		#eliminated Ticket
features_e.append(features.pop(6))		#eliminated Cabin

features_c = []							#to transform in continous from category
features_c.append(features.pop(1))		#sex
features_c.append(features.pop(5))		#embarked

features_t = list(features)
features_t.pop(5)

#transform Age and Embarked which could be NaN into the most representative value
for i in range(m):
	if pd.isnull(dataset.Age[i]):
		dataset.Age[i] = collections.Counter(dataset.Age).most_common()[0][0]
	if pd.isnull(dataset[features_c[1]][i]):
		dataset.Embarked[i] = collections.Counter(dataset.Embarked).most_common()[0][0]
		
############### NOW HAVE TO FORMAT TEST_SET
for i in range(m_t):
	if pd.isnull(test_dataset.Age[i]):
		test_dataset.Age[i] = collections.Counter(test_dataset.Age).most_common()[0][0]
	if pd.isnull(test_dataset.Embarked[i]):
		test_dataset.Embarked[i] = collections.Counter(test_dataset.Embarked).most_common()[0][0]	

to_continous = np.array(dataset[features_c])
remaining_f = np.array(dataset[features])
transf = np.array(np.zeros((m,5)))
#brutal transforming categories in continous features
index = 0
for i in range(len(features_c)):
	for unique in np.unique(to_continous[:,i]):
		for j in range(m):
			if to_continous[j,i] == unique:
				transf[j,index] = 1
		index += 1
		
to_continous_t = np.array(test_dataset[features_c])
remaining_t = np.array(test_dataset[features_t])
transf_t = np.array(np.zeros((m_t,5)))
#even for the test set
index2 = 0
for i in range(len(features_c)):
	for unique in np.unique(to_continous_t[:,i]):
		for j in range(m_t):
			if to_continous_t[j,i] == unique:
				transf_t[j,index2] = 1
		index2 += 1	

#new formatted datasets
dataset_f = np.hstack((transf,remaining_f))
test_dataset_f = np.hstack((transf_t,remaining_t))
print test_dataset_f

X = dataset_f[:,:-1]		#features
Y = dataset_f[:,-1:]		#output

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X, Y)

#Test
x = [  1., 0., 0., 0., 1., 3., 26., 0., 0., 7.925]
print '## TEST ## The class of', x, 'is', decision_tree.predict([x]), 'and should be 1'

#we found another NaN in position 152,9 -> we set his value to 0
test_dataset_f[152][9] = 0

print(pd.DataFrame(test_dataset_f))

print '\nPredictions for test set: '
predictions = decision_tree.predict(test_dataset_f)
print predictions

# open a new file so I can write to it. 
predictions_file = open("mySklearnModelPredictions.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])

for x in range(m_t):
	predictions_file_object.writerow([test_dataset.PassengerId[x], int(predictions[x])])

# Close out the files
predictions_file.close()















