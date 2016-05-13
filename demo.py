#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split
from copy import copy
from time import time

import acoc
import utils
from utils import data_manager, generate_folder_name
import acoc.polygon
from config import CLASSIFIER_CONFIG, SAVE_DIR


CLASSIFIER_CONFIG.plot = False
CLASSIFIER_CONFIG.save = False
CLASSIFIER_CONFIG.training_test_split = True

CLASSIFIER_CONFIG.data_set = 'iris'
SAVE_FOLDER = generate_folder_name(CLASSIFIER_CONFIG.data_set, SAVE_DIR)


def run(**kwargs):
    conf = copy(CLASSIFIER_CONFIG)
    for k, v in kwargs.items():
        conf[k] = v

    data_set = data_manager.load_data_set(conf.data_set)
    X = data_set.data
    y = data_set.target
    class_indices = list(set(y))

    # Split data into training and testing set
    if conf.training_test_split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    else:
        X_train = X_test = X
        y_train = y_test = y

    clf = acoc.PolyACO(X.shape[1], class_indices, save_folder=SAVE_FOLDER)
    clf.train(X_train, y_train, start_time=conf.start_time)
    predictions = clf.evaluate(X_test)
    return acoc.compute_score(predictions, y_test)


if __name__ == "__main__":
    utils.save_dict(CLASSIFIER_CONFIG, parent_folder=SAVE_FOLDER, file_name='config.json')
    scores = []
    runs = 1
    result_str = ''
    start_time = time()
    for i in range(runs):
        scores.append(run(start_time=start_time))
        print("\nRun {}/{} score: {:.4f}".format(i + 1, runs, scores[-1]))
    result_str = ','.join([str(s) for s in scores]) + "\nAverage score with {}-fold cross validation: {:.5f}".format(runs, sum(scores) / runs)
    utils.save_string_to_file(result_str, parent_folder=SAVE_FOLDER, file_name='result.txt')
    print("\n" + result_str)
