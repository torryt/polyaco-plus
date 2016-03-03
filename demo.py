#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from copy import copy

import acoc
import acoc.polygon
from config import CLASSIFIER_CONFIG

SAVE_FOLDER = datetime.utcnow().strftime('%Y-%m-%d_%H%M')

CLASSIFIER_CONFIG.plot = False
CLASSIFIER_CONFIG.save = True


def run(**kwargs):
    conf = copy(CLASSIFIER_CONFIG)
    for k, v in kwargs.items():
        conf[k] = v

    # Loads a sample data set from a pickle file.
    data_set = datasets.load_iris()

    X = data_set.data
    y = data_set.target
    class_indices = list(set(y))

    # Split data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    clf = acoc.PolyACO(X.shape[1], class_indices, conf, SAVE_FOLDER)
    clf.train(X_train, y_train)
    predictions = clf.evaluate(X_test)

    return acoc.compute_score(predictions, y_test)


if __name__ == "__main__":
    scores = []
    runs = 10
    for i in range(runs):
        print("\nRun {}/{}".format(i + 1, runs))
        scores.append(run())
    print("Average score with {}-fold cross validation: {:.2f}".format(runs, sum(scores) / runs))
