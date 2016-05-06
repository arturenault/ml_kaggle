import csv
import time

import sys
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    BaggingClassifier
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures


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

        labeler = LabelEncoder()

        columns = len(x[0])

        for column in range(columns):
            current = [row[column] for row in x]
            labeler.fit(current)
            new_column = labeler.transform(current)
            for row in range(len(x)):
                x[row][column] = new_column[row]

        poly = PolynomialFeatures()
        x = poly.fit_transform(x)

        print(len(x))
        print(len(x[0]))

        x_train = x[:train_size]
        x_test = x[train_size:]

        return x_train, x_test, y


def train(data, labels):
    """
    classifier = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=400, n_jobs=-1)),
        ('ada', AdaBoostClassifier(n_estimators=50,
                                   base_estimator=RandomForestClassifier(
                                       n_estimators=40, n_jobs=-1))),
        ('nc', NearestCentroid())
    ])
    """
    classifier = BaggingClassifier(base_estimator=AdaBoostClassifier(
        base_estimator=RandomForestClassifier(n_estimators=40, n_jobs=-1)),
                                   n_jobs=-1)

    classifier.fit(data, labels)
    return classifier


def classify(classifier, test_data):
    predictions = classifier.predict(test_data)
    predictions = [int(i) for i in predictions]
    return predictions


def output(predictions, filename):
    with open(filename, "w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Id", "Prediction"])
        for i in range(len(predictions)):
            writer.writerow([str(i + 1), predictions[i]])


def main():
    print "Current time: " + time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.time()
    try:
        datafile = sys.argv[1]
        quizfile = sys.argv[2]
        outputfile = sys.argv[3]
    except:
        sys.exit(
            "usage: python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE")

    print "getting data"
    train_data, test_data, labels = get_data(datafile, quizfile)

    train_number = int(len(train_data) * (1 - test_fraction))

    print "training"
    classifier = train(train_data[:train_number], labels[:train_number])

    if test_fraction > 0:
        print "testing"
        score = classifier.score(train_data[train_number:],
                                 labels[train_number:])

        print "Score: " + str(score)

    print "classifying"
    predictions = classify(classifier, test_data)

    print "outputting"
    output(predictions, outputfile)

    end = time.time()
    print "total running time: " + str(end - start)

    # make a noise so you know it's done
    print '\a'


if __name__ == "__main__":
    main()