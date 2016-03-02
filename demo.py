#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from sklearn import datasets
from copy import copy
import numpy as np

import acoc
import acoc.polygon
from config import CLASSIFIER_CONFIG

SAVE = False
SAVE_PHEROMONES_AND_BEST_PATHS = False
SAVE_FOLDER = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
SHOW_PLOT = False

CLASSIFIER_CONFIG.plot = SAVE_PHEROMONES_AND_BEST_PATHS
CLASSIFIER_CONFIG.run_time = 0.1


def run(**kwargs):
    conf = copy(CLASSIFIER_CONFIG)
    for k, v in kwargs.items():
        conf[k] = v

    # Loads a sample data set from a pickle file.
    iris = datasets.load_iris()

    # Use only data samples from 2 out of 3 classes
    data = iris.data[:100]
    target = iris.target[:100]

    clf = acoc.PolyACO(data.shape[1], conf, SAVE_FOLDER)
    clf.train(data, target)
    results = clf.evaluate(data)

    classification_score = (np.equal(results, target).sum() / len(results)) * 100
    return classification_score



if __name__ == "__main__":
    # run()
    for _ in range(1):
        print("\nFinal classification score: {}%".format(run()))
