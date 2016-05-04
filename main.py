
import csv
from os.path import isfile
import sys

from sklearn.ensemble import AdaBoostClassifier

test_fraction = 0.2

def get_data(train_filename, test_filename):
  with open(train_filename) as train_file, open(test_filename) as test_file:
    reader_train = csv.reader(train_file)
    reader_test = csv.reader(test_file)

    reader_train.next()
    reader_test.next()

    x_train = []
    y = []

    x_test = []

    for row in reader_train:
      x_train.append(row[:-1])
      y.append(int(row[-1]))

    for row in reader_test:
      x_test.append(row)


    train_size = len(x_train)

    x = x_train + x_test

    labeler = preprocessing.LabelEncoder()

    columns = len(x[0])

    for column in range(columns):
      current = [row[column] for row in x]
      labeler.fit(current)
      new_column = le.transform(current)
      for row in range(len(x)):
        x[row][column] = new_column[row]

    onehot = OneHotEncoder()
    onehot.fit(x)

    x = onehot.transform(x).toarray()

    x_train = x[train_size:]
    x_test = x[:train_size]

    return x_train, x_test, y

def train(data, labels):
  classifier = AdaBoostClassifier()
  classifier = classifier.fit(data, labels)
  return classifier

def classify(classifier, test_data):
  predictions = classifier.predict(test_data)
  predictions = [int(i) for i in predictions]
  return predictions

def test(classifier, test_data, labels):
  predictions = classify(classifier, test_data)
  correct = 0.0
  for i, prediction in enumerate(predictions):
    if prediction == labels[i]:
      correct += 1
  return correct / len(labels)

def output(predictions, filename):
  with open(filename, "w") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["Id", "Prediction"])
    for i in range(len(predictions)):
      writer.writerow([str(i + 1), predictions[i]])

def main():
  try:
    datafile = sys.argv[1]
    quizfile = sys.argv[2]
    outputfile = sys.argv[3]
  except:
    sys.exit("usage: python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE")

  print "getting data"
  train_data, test_data, labels = get_train_data(datafile, quizfile)

  train_number = int(len(train_data) * (1 - test_fraction))
  test_number = len(train_data) - train_number

  print "training"
  classifier = train(train_data.head(train_number), labels[:train_number])

  print "testing"
  score = test(classifier, train_data.tail(test_number), labels[train_number:])

  print "Score: " + str(score)

  print "classifying"
  predictions = classify(classifier, test_data)

  print "outputting"
  output(predictions, outputfile)



if __name__=="__main__":
  main()
