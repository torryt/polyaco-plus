import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
import os.path as osp

import acoc
import utils
from acoc import acoc_plotter
from utils import utils

from config import SAVE_DIR

CONFIG = {
    'ant_count': 2000,
    'number_runs': 50,
    'q': 5.0,
    'q_min': 0.1,
    'q_max': 20.0,
    'q_init': 0.1,
    'rho': 0.02,
    'alpha': 1,
    'beta': 0.05,
    'ant_init': 'weighted',
    'decay_type': 'random_type'
}


def run(*args):
    config = dict(CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]
    if 'data_set' in config:
        data = pickle.load(open('../data_sets.pickle', 'rb'), encoding='latin1')[config['data_set']]
    else:
        data = pickle.load(open('../data_sets.pickle', 'rb'), encoding='latin1')['rectangle']
    number_runs = config['number_runs']
    clf = acoc.Classifier(config)
    all_ant_scores = np.zeros((number_runs, config['ant_count']))

    for i in range(number_runs):
        iter_string = "Iteration: {}/{}".format(i + 1, number_runs)
        ant_scores, path = \
            clf.classify(data, False, ', ' + iter_string)
        utils.print_on_current_line(iter_string)
        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0)


def parameter_tester(parameter_name, values, config=CONFIG, data_set='rectangle'):
    save_folder = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    save_folder = osp.basename(full_path)
    config['data_set'] = data_set
    print("\n\nExperiment for parameter '{}' with values {}".format(parameter_name, values))
    plt.clf()
    all_scores = []
    for index, v in enumerate(values):
        print("Run {} with value {}".format(index+1, v))
        scores = run((parameter_name, v), ('data_set', data_set))
        all_scores.append(scores)
        utils.print_on_current_line('')

    utils.save_dict(config, 'config_' + parameter_name + '.txt', save_folder)
    utils.save_object(all_scores, 'data', save_folder)
    labels = [parameter_name + '=' + str(v) for v in values]
    f1 = acoc_plotter.plot_curves(all_scores, labels)
    acoc_plotter.save_plot(f1, save_folder)
    # f2 = acoc_plotter.plot_smooth_curves(all_scores, labels)
    # acoc_plotter.save_plot(f2, SAVE_FOLDER)

if __name__ == "__main__":
    # parameter_tester('ant_init', ['random', 'static', 'weighted', 'on_global_best', 'chance_of_global_best'])
    parameter_tester('decay_type', ['random_type', 'grad_type'])
    # parameter_tester('q_init', [CONFIG['q_max'], CONFIG['q_min']])
    # parameter_tester('rho', [0.001, 0.01, 0.02, 0.1, 0.3])
    # parameter_tester('iterations', [1, 2, 5, 10])
