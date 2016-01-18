import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
import os.path as osp

import acoc
from acoc import acoc_plotter
import utils

from config import SAVE_DIR

CONFIG = {
    'ant_count':    3000,
    'number_runs':  50,
    'tau_min':      0.001,
    'tau_max':      1.0,
    'tau_init':     0.001,
    'rho':          0.02,
    'alpha':        1,
    'beta':         0.1,
    'ant_init':     'weighted',
    'decay_type':   'probabilistic',
    'data_set':     'rectangle_small'
}


def run(*args):
    config = dict(CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]
    data = pickle.load(open('../utils/data_sets.pickle', 'rb'), encoding='latin1')[config['data_set']]
    number_runs = config['number_runs']

    clf = acoc.Classifier(config)
    all_ant_scores = np.zeros((number_runs, config['ant_count']))
    all_polygons = np.zeros((number_runs, config['ant_count']))
    all_costs = np.zeros((number_runs, config['ant_count']))

    for i in range(number_runs):
        iter_string = "Iteration: {}/{}".format(i + 1, number_runs)
        ant_scores, path = \
            clf.classify(data, False, ', ' + iter_string)
        utils.print_on_current_line(iter_string)
        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0), all_polygons.mean(0), all_costs.mean(0)


def parameter_tester(parameter_name, values, config=CONFIG):
    save_folder = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    save_folder = osp.basename(full_path)
    print("\n\nExperiment for parameter '{}' with values {}".format(parameter_name, values))

    plt.clf()
    all_scores = []
    all_costs = []
    all_polygon_lengths = []
    for index, v in enumerate(values):
        print("Run {} with value {}".format(index+1, v))
        scores, polygons, costs = run((parameter_name, v))
        all_scores.append(scores)
        all_costs.append(costs)
        all_polygon_lengths.append(polygons)
        utils.print_on_current_line('')
    utils.save_dict(config, save_folder, 'config_' + parameter_name + '.txt')
    labels = [parameter_name + '=' + str(v) for v in values]

    def save_results(data, typ, ylabel):
        utils.save_object(data, save_folder, 'data_' + typ)
        fig = acoc_plotter.plot_curves(data, labels, ylabel)
        acoc_plotter.save_plot(fig, save_folder, 'noise_' + typ)
        acoc_plotter.save_plot(acoc_plotter.plot_smooth_curves(data, labels, ylabel), save_folder, 'smooth_' + typ)

    results = [(all_scores, 'score', 'Score'),
               (all_costs, 'cost', 'Cost'),
               (all_polygon_lengths, 'length', 'Length of Polygon')]
    [save_results(*obj) for obj in results]

if __name__ == "__main__":

    # parameter_tester('tau_min', [0.001, 0.01, 0.1])
    # parameter_tester('tau_max', [0.1, 1.0, 10.0, 100.0])
    parameter_tester('beta', [0, 0.01, 0.1, 1.0])

    # parameter_tester('ant_init', ['random', 'static', 'weighted', 'on_global_best', 'chance_of_global_best'])
    # parameter_tester('ant_init', ['random', 'static', 'weighted', 'on_global_best', 'chance_of_global_best'])
    # parameter_tester('decay_type', ['probabilistic', 'gradual'])
    # parameter_tester('tau_init', [CONFIG['tau_max'], CONFIG['tau_min']])
    # parameter_tester('rho', [0.001, 0.01, 0.02, 0.1, 0.3])
    # parameter_tester('iterations', [1, 2, 5, 10])
