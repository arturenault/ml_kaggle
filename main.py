from sklearn import tree
import numpy
import pandas
import csv

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
    x = []
    y = []
    reader.next()
    for row in reader:
      x.append([(float(i) if is_num(i) else i) for i in row])
      y.append(float(row[-1]))

    dataframe = pandas.DataFrame(data=x)

    processed_data = pandas.get_dummies(dataframe)

    matrix = processed_data.as_matrix()

    return matrix, y

def train(data, labels):
  classifier = tree.DecisionTreeClassifier()
  classifier = classifier.fit(data, labels)
  return classifier

def get_test_data():
  with open(test_filename) as test_file:
    reader = csv.reader(test_file)
    x = []
    reader.next()
    for row in reader:
      x.append([(float(i) if is_num(i) else i )for i in row])

    dataframe = pandas.DataFrame(data=x)

    processed_data = pandas.get_dummies(dataframe)

    matrix = processed_data.as_matrix()

    return matrix

def classify(classifier, test_data):
  predictions = []
  for datum in test_data:
      predictions.append(classifier.predict(datum))
  return predictions

def output(predictions):
  with open(output_filename) as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["id", "Prediction"])
    for i in range(len(predictions)):
      writer.writerow([str(i + 1), predictions[i]])


def main():
  data, labels = get_train_data()

  classifier = train(data, labels)

  test_data = get_test_data()

  predictions = classify(classifier, test_data)

  output(predictions)



if __name__=="__main__":
  main()
