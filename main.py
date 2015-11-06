#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import acoc
from utils import utils
from acoc import acoc_plotter as plotter
import utils.data_generator as dg
import pickle


live_plot = True
save = True
show_plot = True
iterations = 5

clf_config = {
    'ant_count':    200,
    'q':            5.0,
    'q_min':        0.1,
    'q_max':        20.0,
    'q_init':       20.0,
    'rho':          0.02,
    'alpha':        1,
    'beta':         0.05,
    'ant_init':     'weighted',
    'decay_type':   'grad_type'
}

clf = acoc.Classifier(clf_config)
data_sets = pickle.load(open('data_sets.pickle', 'rb'), encoding='latin1')


def run():
    all_ant_scores = np.zeros((iterations, clf.ant_count))
    global_best_polygon = []
    global_best_score = 0

    # red = dg.uniform_circle(3.0, 500, 1)
    # blue = dg.uniform_circle(2.0, 500, 0)
    red = dg.uniform_rectangle((1, 3), (2, 4), 500, 0)
    blue = dg.uniform_rectangle((4, 6), (2, 4), 500, 1)
    data = np.concatenate((red, blue), axis=1)

    for i in range(iterations):
        iter_string = "Iteration: {}/{}".format(i + 1, iterations)
        ant_scores, path = \
            clf.classify(data, live_plot, ', ' + iter_string)
        utils.print_on_current_line(iter_string)
        print(", Best ant score: {}".format(max(ant_scores)))

        all_ant_scores[i, :] = ant_scores
        if max(ant_scores) > global_best_score:
            global_best_polygon = path
            global_best_score = max(ant_scores)

    score = acoc.polygon_score(global_best_polygon, data)
    if save:
        utils.save_object(all_ant_scores.mean(0), file_name='scores')
        # TODO: Save best polygon without recursion. Just points maybe?
        # utils.save_object(global_best_polygon, file_name='best_path')
        utils.save_dict(clf_config, 'config.txt')
    print("\n\nGlobal best score(points) {}".format(score))
    print("Global best score(|solution| and points): {}".format(global_best_score))

    plotter.plot_path_with_data(global_best_polygon, data, save=save, show=show_plot)
    plotter.plot_ant_scores(all_ant_scores.mean(0), save=save, show=show_plot)


run()
