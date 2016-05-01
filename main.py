import numpy
import pandas
import csv
import argparse
from os.path import isfile
import pickle

from sklearn.ensemble import AdaBoostClassifier

train_filename = "data.csv"
test_filename  = "quiz.csv"
output_filename = "output.csv"

def is_num(string):
  try:
    float(string)
    return True
  except ValueError:
    return False

def get_train_data():
  with open(train_filename) as train_file:
    reader = csv.reader(train_file)

    series = reader.next()
    x = dict((i, []) for i in series[:-1])
    y = []

    for row in reader:
      for i, element in enumerate(row[:-1]):
        if is_num(element):
          x[series[i]].append(float(element))
        else:
          x[series[i]].append(element)
      y.append(float(row[-1]))


    dataframe = pandas.DataFrame(data=x)

    processed_data = pandas.get_dummies(dataframe)

    return processed_data, y

def train(data, labels):
  classifier = AdaBoostClassifier()
  classifier = classifier.fit(data, labels)
  return classifier

def get_test_data():
  with open(test_filename) as test_file:
    reader = csv.reader(test_file)

    series = reader.next()
    x = dict((i, []) for i in series)

    for row in reader:
      for i, element in enumerate(row):
        if is_num(element):
          x[series[i]].append(float(element))
        else:
          x[series[i]].append(element)

    dataframe = pandas.DataFrame(data=x)

    processed_data = pandas.get_dummies(dataframe)

    return processed_data

def fix_test_data(train_data, test_data):
  for series in train_data:
    if series not in test_data:
      test_data[series] = 0.0

  for series in test_data:
    if series not in train_data:
      test_data.drop(series, axis=1, inplace=True)

def classify(classifier, test_data):
  predictions = classifier.predict(test_data)
  predictions = [int(i) for i in predictions]
  return predictions

def output(predictions):
  with open(output_filename, "w") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["id", "Prediction"])
    for i in range(len(predictions)):
      writer.writerow([str(i + 1), predictions[i]])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", "--regenerate", action="store_true")
  args = parser.parse_args()

  print "getting training data"
  train_data, labels = get_train_data()

  print "training"
  classifier = train(train_data, labels)

  if isfile("test_data.pickle") and not args.regenerate:
    print "loading test data"
    test_data = pickle.load(open("test_data.pickle"))

  else:
    print "getting test data"
    test_data = get_test_data()

    print "fixing test data"
    fix_test_data(train_data, test_data)


  print "classifying"
  predictions = classify(classifier, test_data)

  print "outputting"
  output(predictions)



if __name__=="__main__":
  main()
