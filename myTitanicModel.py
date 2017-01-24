#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# My first model for Titanic Problem
#
# Tring to predict who died on the Titanic and
# and who lived throught a decision tree
#
# Data needed:  train.csv 
#				test.csv
#
# Esecution:
#    python2.7 myTitanicModel.py
# or
#    chmod 755 myTitanicModel.py   <--- tantum
#    ./myTitanicModel.py

##################################################
#
# Modules
# Reading and Manipulation of the dataset
import pandas as pd

# Numerics Methods
import numpy as np

# Pretty-printing of data structures
import pprint

# Counting and Enumeration of values (using for calculate the frequency of inputs)
import collections

import math

######################################
#
# Parameters

# Minimum entropy that a dataset need to have to divide it
entropy_threshold = 0.001

##########################################
# Functions

# Calculate Entropy: given a vector of objects(numerics or strings),
# estimate the entropy of the Casual Variable where it comes from
def entropy(y):
	count = collections.Counter(y) 			# count how many times each value is present in y
	s = 0									# for cumulate the sum
	m = float(len(y))						# divisor for calculating frequencies
	
	# now for every value present in the vector
	for c in count.values():
		p = c/m								# estimate the probability of that value
		s -= p * np.log2(p)					# add the entropy to the total entropy 
	
	return s								# return the total entropy found

# Testing the entropy function: the following call is made on a vector with 4 diffrents values with frequency 1/2, 1/4, 1/8, 1/8
# and the function should return entropy = 7/4
print 'Entropy test:', entropy([1,'a',1,'pippo',1,'a',4.7,1]), 'should be', 7.0/4.0

# Partition of a dataset D in 2 subpartitions d1 and d2 such us to minimize the expected entropy
# The partition is based on a condition of the form "x[j]<theta", and seek for the optimal values of j and theta
# The second parameter is the OUTPUT column which should be excluded
def find_split(D):
	H_min = 1.0e100							# remember the minimum entropy
	m,n = D.shape							# get the numbers of features and samples
	
	for j in range(n):						# exclude output column
		for theta in np.unique(D[[j]]):			# iterate on all possible values of that column
			d1 = D[ D[j]<theta ]			# put in d1 the samples for which x[j]<theta and in d2 the remainings
			d2 = D[ D[j]>=theta ]
			
			H1 = entropy(d1[n-1])			# calculate entropy of the splitting (excluding the output column)
			H2 = entropy(d2[n-1])
			
			H_mean = ( (H1*len(d1)) + (H2*len(d2)) ) / m	# calculate the Average Entropy
			
			if H_mean < H_min :				# if we found a lower entropy we store the minimum parameters and minimum entropy
				H_min = H_mean
				j_min = j
				theta_min = theta
				d1_min = d1
				d2_min = d2
				H1_min = H1
				H2_min = H2
	return j_min, theta_min, d1_min, d2_min, H1_min, H2_min		#return the minimum parameters
	
# Recursive construction of the tree starting from dataset D
# We indicate with y the output feature
def build_tree(D):
	m, n = D.shape								# get the number of features and samples
	j, theta, d1, d2, H1, H2 = find_split(D)	# find the ottimal parameters for constructing the tree
	
	print 'x[',j,'] <',theta
	print 'H1, H2 =', H1, H2
	print 'd1 output = ', d1[[11]]
	print 'd2 output =', d2[[11]]

	# left subtree
	if H1 < entropy_threshold:					# if we found an entropy less than our threshold it means we found a leaf
		left = collections.Counter(d1[n-1]).most_common()[0][0]	# we say that the most common value in this leaf is the rappresentant of this subtree
	else:										# otherwise it means that we have to split again
		left = build_tree(d1)
		
	# right subtree (as for the left one)
	if H2 < entropy_threshold:
		right = collections.Counter(d2[n-1]).most_common(1)[0][0]
	else:
		right = build_tree(d2)
	
	return j, theta, left, right
	
# Classifing a vector of features x, using the tree 'tree', indicating the output feature y
# Note that the function is tail_recursive (we can write it using a while cycle whithout using stacks)
def classify(x,y,tree):
	if isinstance(tree, tuple):					# if the tree is a tuple it means that it isn't a leaf so we have to visit it deeper
		j, theta, left, right = tree			# we extract the parameters
		return classify(x, y, left if x[j]<theta else right)	# we call recursively classify(...) on the left or on the right tree based on the condition 'x[j]<theta'
	else:
		return tree								# if the tree is a leaf it means that it is contains the value to be predicted

# Moving the 2nd column(which is the output to predict) at the end of the DataFrame
def move_output_to_end(D, col):
    m, n = D.shape
    cols = range(n)
	
    if col>=n or col<0:
        return
    if col is 0:
        cols = cols[1:n] + cols[0:1]
    else:
        cols = cols[0:col] + cols[col+1:n+1] + cols[col:col+1]
    
    return pd.DataFrame(D)

# Trasforming categoric features into continous
def categoric_to_continuous(D):
	m,n = D.shape
	index1 = -1

	conCols = [1, 5, 6, 7, 9]	
	catCols = [2, 4, 11]
	outCols = [0, 3, 8, 10]
	
	print 'Eliminating some features...', outCols

	D = np.array(D)
	#brutal transforming NaN in 0(zeros)
	for x in range(m):
		for y in range(n):
			if pd.isnull(D[x,y]):
				D[x,y]=0
	D = pd.DataFrame(D)

	newDataFrame = D[conCols]	
	oldDataFrame = D[catCols]
	old = np.array(oldDataFrame)
	continouzedFrame = np.array(np.zeros((m,11)))
	D = np.array(D)
	
	newDataFrame = np.array(newDataFrame)

	print 'Trasforming categorical vars to continous...'
	for i in catCols:
		values = np.unique(D[:,i])
		print 'Casual Var: ', values
		for val in values:
			index1 += 1
			print 'Trasforming: ', val
			for j in range(m):
				if D[j,i] is val:
					continouzedFrame[j,index1] = 1 

	
	new = pd.DataFrame(np.hstack((newDataFrame, continouzedFrame)))
	print 'Transforming DONE'
	print 'New dataset shape', new.shape
	return new
	
	

###################################################
#
# Main Program
#
# Training Set Reading
train = pd.read_csv('train.csv', header=1)
# Test Set Reading
test = pd.read_csv('test.csv', header=1)			
		
train = categoric_to_continuous(train)
train = move_output_to_end(train, 0)

m,n = train.shape

#print 'Test: ', test

#prediction_file = open("firstGenderModel.csv", "wb")		#open a pointer to a new file so we can write the predicted results on it
#prediction_file_object = csv.writer(prediction_file)

m, n = train.shape
print 'Output entropy:', entropy(train)				# total entropy of training set

decision_tree = build_tree(train)
print 'DecisionTree: '							# construct the decision tree
pprint.pprint (decision_tree)								# print the tree



















