from collections import Counter
import csv
import pdb
from operator import itemgetter
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import tree
import time

train_filename = "data.csv"
test_filename = "quiz.csv"
output_filename = 'output_2.csv'

def train():
	t0 = time.time()
	print t0
	with open(train_filename) as train_file, open(test_filename) as test_file, open(output_filename, 'w') as out_file:
		reader1 = csv.reader(train_file)
		reader2 = csv.reader(test_file)
		series = reader1.next()
		series2 = reader2.next
		x_train = []
		y_train = []

		x_test = []

		for row in reader1:
			x_train.append(row[:-1])
			y_train.append(float(row[-1]))

		for row in reader2:
			x_test.append(row)

		size = len(x_train)
		
		x = x_train + x_test
		le = preprocessing.LabelEncoder()
		column_count = len(x[0])
		for i in range(0,column_count):
			current_col = [row[i] for row in x]
			le.fit(current_col)
			new_column = le.transform(current_col)
			for index in range(0,len(x)):
				x[index][i] = new_column[index]

		enc = OneHotEncoder()
		enc.fit(x)
		x = enc.transform(x).toarray()

		x_train = x[:size]
		x_test = x[size:]

		# split_number = int(round(size * .8))
		# training_x = x_train[:split_number]
		# test_x = x_train[split_number:]
		# training_y = y_train[:split_number]
		# test_y = y_train[split_number:]
		# pdb.set_trace()

		print "training"
		clf = tree.DecisionTreeClassifier()
		clf.fit(x_train, y_train)
		# neigh = KNeighborsClassifier(n_neighbors=6)
		# neigh.fit(training_x,training_y)
		print "done"

		print "predicting"
		predicted_y = clf.predict(x_test)
		print "done"

		# correct = 0
		# wrong = 0
		# for index in range(0,len(test_x)):
		# 	if predicted_y[index] == test_y[index]:
		# 		correct = correct + 1
		# 	else:
		# 		wrong = wrong + 1
		
		# print "correct " + str(correct)
		# print "wrong " + str(wrong)
 		# print str(float(correct)/float(correct + wrong))
		# print "total " + str(correct + wrong)

		writer = csv.writer(out_file, delimter=',')
		writer.writerow(['Id','Prediction'])
		for index in range(0, len(predicted_y)):
			writer.writerow([index,int(predicted_y[index])])
	t1 = time.time()
	print t1-t0
	
train()






