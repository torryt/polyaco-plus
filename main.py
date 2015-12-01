#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import pickle
from datetime import datetime

import os.path as osp
import acoc
import utils
from utils.data_generator import uniform_circle
from acoc.acoc_matrix import AcocMatrix
from acoc import acoc_plotter as plotter
from timeit import timeit

SAVE = True
SAVE_PHEROMONE_VALUES = False
SAVE_FOLDER = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
SHOW_PLOT = False
NUMBER_RUNS = 1
clf_config = {
    'ant_count':    4000,
    'q':            5.0,
    'q_min':        0.1,
    'q_max':        20,
    'q_init':       20.0,
    'rho':          0.02,
    'alpha':        1,
    'beta':         0.05,
    'ant_init':     'on_global_best',
    'decay_type':   'random_type'
}

clf = acoc.Classifier(clf_config, osp.join(SAVE_FOLDER, 'live_plot'))
data_sets = pickle.load(open('data_sets.pickle', 'rb'), encoding='latin1')
data = data_sets['rectangle_gaussian']


def run():
    all_ant_scores = np.zeros((NUMBER_RUNS, clf.ant_count))
    global_best_polygon = []
    global_best_score = 0

    for i in range(NUMBER_RUNS):
        iter_string = "Iteration: {}/{}".format(i + 1, NUMBER_RUNS)
        ant_scores, path = \
            clf.classify(data, SAVE_PHEROMONE_VALUES, ', ' + iter_string)
        utils.print_on_current_line(iter_string)
        print(", Best ant score: {}".format(max(ant_scores)))

        all_ant_scores[i, :] = ant_scores
        if max(ant_scores) > global_best_score:
            global_best_polygon = path
            global_best_score = max(ant_scores)

    score = clf.polygon_score(global_best_polygon, data)
    if SAVE:
        utils.save_object(all_ant_scores.mean(0), 'scores', SAVE_FOLDER)
        utils.save_dict(clf_config, 'config.txt', SAVE_FOLDER)
    print("\n\nGlobal best score(points) {0:.5f}".format(score))
    print("Global best score(|solution| and points): {}".format(global_best_score))

    matrix = AcocMatrix(data)
    plotter.plot_path_with_data(global_best_polygon, data, matrix, save=SAVE, save_folder=SAVE_FOLDER, show=SHOW_PLOT)
    plotter.plot_ant_scores(all_ant_scores.mean(0), save=SAVE, show=SHOW_PLOT, save_folder=SAVE_FOLDER)


time = timeit('run()', setup='from __main__ import run', number=1)
print("Time pr ant: {}".format(time / clf_config['ant_count']))
# run()
