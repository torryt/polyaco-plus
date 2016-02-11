#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import pickle
from datetime import datetime
from timeit import timeit

import acoc
import utils
from acoc.acoc_matrix import AcocMatrix
from acoc import acoc_plotter as plotter


SAVE = False
SAVE_PHEROMONES_AND_BEST_PATHS = False
SAVE_FOLDER = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
SHOW_PLOT = False
clf_config = {
    'run_time':         10,      # Algorithm runtime in seconds
    'tau_min':          0.001,
    'tau_max':          1.0,
    'tau_init':         0.001,
    'rho':              0.02,
    'alpha':            1,
    'beta':             0.01,
    'ant_init':         'weighted',
    'decay_type':       'probabilistic',
    'gpu':              True,
    'granularity':      10,
    'multi_level':      False,
    'max_level':        None,
    'convergence_rate': 800,
    'data_set':         'semicircle_gaussian'
}


def run(**kwargs):
    conf = dict(clf_config)
    for k, v in kwargs.items():
        conf[k] = v

    clf = acoc.Classifier(conf, SAVE_FOLDER)
    data = pickle.load(open('utils/data_sets.pickle', 'rb'), encoding='latin1')[conf['data_set']]

    ant_scores, polygon = clf.classify(data, SAVE_PHEROMONES_AND_BEST_PATHS)
    print(", Best ant score: {}".format(max(ant_scores)))

    if SAVE:
        utils.save_object(ant_scores, SAVE_FOLDER, 'scores')
        utils.save_object(acoc.polygon_to_array(polygon), SAVE_FOLDER, 'path')
        utils.save_dict(conf, SAVE_FOLDER, 'config.txt')

    if SAVE or SHOW_PLOT:
        matrix = AcocMatrix(data)
        plotter.plot_path_with_data(polygon, data, matrix, save=SAVE, save_folder=SAVE_FOLDER, show=SHOW_PLOT)
        plotter.plot_ant_scores(ant_scores, save=SAVE, show=SHOW_PLOT, save_folder=SAVE_FOLDER)
    return max(ant_scores)

if __name__ == "__main__":
    run()
    # scores = []
    # for _ in range(10):
    #     scores.append(run())
    # print("Average best score: {}".format(sum(scores) / len(scores)))
