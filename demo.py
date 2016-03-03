#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
from sklearn import datasets
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

    # Use only data samples from 2 out of 3 classes
    data = data_set.data
    target = data_set.target
    class_indices = list(set(target))

    clf = acoc.PolyACO(data.shape[1], class_indices, conf, SAVE_FOLDER)
    clf.train(data, target)
    predictions = clf.evaluate(data)

    return acoc.compute_score(predictions, target)


if __name__ == "__main__":
    # run()
    for _ in range(1):
        print("\nFinal classification score: {:.4f}%".format(run()))
