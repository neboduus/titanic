#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import csv as csv

test_file = open('test.csv', 'rb')								#read in the test file by opening a python object to read and another to write
test_file_object = csv.reader(test_file)
header = test_file_object.next()								#skip the header file

prediction_file = open("firstGenderModel.csv", "wb")		#open a pointer to a new file so we can write to it
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       								# For each row in test.csv
    if row[3] == 'female':         								# is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    		# predict 1
    else:                              							# or else if male,       
        prediction_file_object.writerow([row[0],'0'])    		# predict 0
test_file.close()
prediction_file.close()
