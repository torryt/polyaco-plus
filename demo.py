#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from datetime import datetime

import acoc
import acoc.polygon
import utils
from config import CLASSIFIER_CONFIG
from acoc.acoc_matrix import AcocMatrix
from acoc import acoc_plotter as plotter


SAVE = True
SAVE_PHEROMONES_AND_BEST_PATHS = False
SAVE_FOLDER = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
SHOW_PLOT = False


def run(**kwargs):
    conf = dict(CLASSIFIER_CONFIG)
    for k, v in kwargs.items():
        conf[k] = v

    clf = acoc.Classifier(conf, SAVE_FOLDER)
    # Loads a sample data set from a pickle file.
    data = pickle.load(open('utils/data_sets.pickle', 'rb'), encoding='latin1')[conf['data_set']]

    ant_scores, polygon = clf.classify(data, SAVE_PHEROMONES_AND_BEST_PATHS)
    print(", Best ant score: {}".format(max(ant_scores)))

    if SAVE:
        utils.save_object(ant_scores, SAVE_FOLDER, 'scores')
        utils.save_object(acoc.polygon.polygon_to_array(polygon), SAVE_FOLDER, 'path')
        utils.save_dict(conf, SAVE_FOLDER, 'config.txt')

    if SAVE or SHOW_PLOT:
        matrix = AcocMatrix(data)
        plotter.plot_path_with_data(polygon, data, matrix, save=SAVE, save_folder=SAVE_FOLDER, show=SHOW_PLOT)
        plotter.plot_ant_scores(ant_scores, save=SAVE, show=SHOW_PLOT, save_folder=SAVE_FOLDER)
    return max(ant_scores)

if __name__ == "__main__":
    # run()
    scores = []
    for _ in range(1):
        scores.append(run())
    print("Average best score: {}".format(sum(scores) / len(scores)))
