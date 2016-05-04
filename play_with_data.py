from collections import Counter
import csv
import pdb
from operator import itemgetter
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

train_filename = "data.csv"

def main():
	with open(train_filename) as train_file:
		reader = csv.reader(train_file)
		series = reader.next()
		x = []
		y = []

		for row in reader:
			x.append(row[:-1])
			y.append(float(row[-1]))

		size = len(x)
		
		new_x = []
		le = preprocessing.LabelEncoder()
		column_count = len(x[0])
		for i in range(0,column_count):
			current_col = [row[i] for row in x]
			le.fit(current_col)
			new_column = le.transform(current_col)
			for index in range(0,size):
				x[index][i] = new_column[index]

		# enc = OneHotEncoder()
		# x = enc.fit_transform(x)
		# pdb.set_trace()

		split_number = int(round(size * .8))
		training_x = x[:split_number]
		test_x = x[split_number:]
		training_y = y[:split_number]
		test_y = y[split_number:]

		print "training"
		neigh = KNeighborsClassifier()
		neigh.fit(training_x,training_y)
		print "done"

		strong = []
		percentage_used = .95
		for i in range(0,52):
			column_values_one = []
			column_values_neg = []
			for j in range(0,len(training_x)):
				if y[j] == -1:
					column_values_neg.append(training_x[j][i])
				elif y[j] == 1:
					column_values_one.append(training_x[j][i])
			count_one = Counter(column_values_one)
			count_neg = Counter(column_values_neg)
			for element in set(list(count_one.elements())):
				value_one = count_one[element]
				value_neg = count_neg[element]
				percentage = float(value_one)/float(value_one+value_neg)
				if (percentage >= percentage_used or percentage <= 1-percentage_used) and value_one + value_neg > 49:
					new_strong = {}
					new_strong["column"] = i
					new_strong["value"] = element
					if percentage >= percentage_used:
						new_strong["label"] = 1
						new_strong["percentage"] = percentage
					else:
						new_strong["label"] = -1
						new_strong["percentage"] = 1 - percentage
					new_strong["appearances"] = value_one + value_neg
					strong.append(new_strong)
		sorted_strong = sorted(strong, key=itemgetter('percentage'), reverse=True)
		length = len(sorted_strong)
		last = sorted_strong[length-1]

		correct = 0
		wrong = 0
		not_finish = 0
		total = len(test_x)
		for i in range(0,len(test_x)):
			for classifier in sorted_strong:
				if test_x[i][classifier['column']] == classifier['value']:
					if test_y[i] == classifier['label']:
						correct = correct + 1
						break
					else:
						wrong = wrong + 1
						break
				if classifier == last:
					predicted_y = neigh.predict([test_x[i]])
					if test_y[i] == predicted_y[0]:
						correct = correct + 1
					else:
						wrong = wrong + 1
					# not_finish = not_finish + 1
		print "correct " + str(correct)
		print "wrong " + str(wrong)
		print "didnt find " + str(not_finish)
 		print str(float(correct)/float(correct + wrong))
 		print str(float(correct + wrong)/float(correct + wrong + not_finish))
		print "total " + str(total) + " " + str(correct + wrong + not_finish)

main()