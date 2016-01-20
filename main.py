#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import pickle
from datetime import datetime
import os.path as osp
from timeit import timeit

import os.path as osp
import acoc
import utils
from acoc.acoc_matrix import AcocMatrix
from acoc import acoc_plotter as plotter
from timeit import timeit

SAVE = False
SAVE_PHEROMONE_VALUES = False
SAVE_FOLDER = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
SHOW_PLOT = False
clf_config = {
    'ant_count':    1500,
    'tau_min':      0.001,
    'tau_max':      1.0,
    'tau_init':     0.001,
    'rho':          0.02,
    'alpha':        1,
    'beta':         0.01,
    'ant_init':     'weighted',
    'decay_type':   'probabilistic',
    'gpu':          False,
    'granularity':  10
}

clf = acoc.Classifier(clf_config, osp.join(SAVE_FOLDER, 'live_plot'))
data = pickle.load(open('utils/data_sets.pickle', 'rb'), encoding='latin1')['rectangle']


def run():
    ant_scores, polygon, _ = clf.classify(data, SAVE_PHEROMONE_VALUES)
    print(", Best ant score: {}".format(max(ant_scores)))

    if SAVE:
        utils.save_object(ant_scores, SAVE_FOLDER, 'scores')
        utils.save_object(acoc.polygon_to_array(polygon), SAVE_FOLDER, 'path')
        utils.save_dict(clf_config, SAVE_FOLDER, 'config.txt')

    if SAVE or SHOW_PLOT:
        matrix = AcocMatrix(data)
        plotter.plot_path_with_data(polygon, data, matrix, save=SAVE, save_folder=SAVE_FOLDER, show=SHOW_PLOT)
        plotter.plot_ant_scores(ant_scores, save=SAVE, show=SHOW_PLOT, save_folder=SAVE_FOLDER)

if __name__ == "__main__":
    runs = 10
    cpu_time = timeit('run()', setup='from __main__ import run', number=runs)

    print("Total runtime (averaged over {} runs): {:.6f} seconds\n\n".format(runs, cpu_time / runs))
