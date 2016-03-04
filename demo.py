#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from sklearn.cross_validation import train_test_split
from copy import copy

import acoc
import utils
from utils import data_manager
import acoc.polygon
from config import CLASSIFIER_CONFIG

SAVE_FOLDER = datetime.utcnow().strftime('%Y-%m-%d_%H%M')

CLASSIFIER_CONFIG.plot = False
CLASSIFIER_CONFIG.save = True
CLASSIFIER_CONFIG.data_set = 'iris'


def run(**kwargs):
    conf = copy(CLASSIFIER_CONFIG)
    for k, v in kwargs.items():
        conf[k] = v

    ####
    # Loads a sample data set from a pickle file.
    ####
    if conf.data_set == 'iris':
        data_set = data_manager.load_iris()
    else:
        CLASSIFIER_CONFIG.data_set = 'breast_cancer'
        data_set = data_manager.load_breast_cancer()

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
    result_str = ''
    for i in range(runs):
        scores.append(run())
        print("\nRun {}/{} score: {:.4f}".format(i + 1, runs, scores[-1]))
    utils.save_dict(CLASSIFIER_CONFIG, parent_folder=SAVE_FOLDER, file_name='config.json')
    result_str = ','.join([str(s) for s in scores]) + "\nAverage score with {}-fold cross validation: {:.5f}".format(runs, sum(scores) / runs)
    utils.save_string_to_file(result_str, parent_folder=SAVE_FOLDER, file_name='result.txt')
    print("\n" + result_str)
