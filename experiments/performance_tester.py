import numpy as np
import pickle
from datetime import datetime
import os.path as osp
import time

import acoc
from acoc import acoc_plotter
import utils

from config import SAVE_DIR

CONFIG = {
    'ant_count':    100,
    'number_runs':  1,
    'tau_min':      0.001,
    'tau_max':      1.0,
    'tau_init':     0.001,
    'rho':          0.02,
    'alpha':        1,
    'beta':         0.1,
    'ant_init':     'weighted',
    'decay_type':   'probabilistic',
    'data_set':     'rectangle',
    'granularity':  10,
    'gpu':          False
}


def run(*args):
    config = dict(CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]
    data = pickle.load(open('../utils/data_sets.pickle', 'rb'), encoding='latin1')[config['data_set']]
    number_runs = config['number_runs']

    clf = acoc.PolyACO(config)
    dropped = 0

    run_times = np.zeros(number_runs, dtype=float)
    for i in range(number_runs):
        iter_string = "Iteration: {}/{}".format(i + 1, number_runs)

        start = time.clock()
        a_score, best_poly, dropped = clf.train(data)
        end = time.clock()
        run_times[i] = end - start

        utils.print_on_current_line(iter_string)
    return np.mean(run_times), dropped


def generate_folder_name():
    now = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, now) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, now) + '-' + str(iterator)
    return osp.basename(full_path)


def performance(parameter_name, values, config=CONFIG):
    save_folder = generate_folder_name()
    print("\n\nPerformance test for parameter '{}' with values {}".format(parameter_name, values))

    results = np.empty((len(values), 2), dtype=float)
    drop = np.empty((len(values), 2), dtype=float)

    for index, v in enumerate(values):
        print("Run {} with value {} on GPU".format(index+1, v))
        results[index, 0], drop[index, 0] = run((parameter_name, v), ('gpu', True))
        utils.clear_current_line()

        # print("Run {} with value {} on CPU".format(index+1, v))
        # results[index, 1], drop[index, 1] = run((parameter_name, v), ('gpu', False))
        # utils.clear_current_line()

    # print("Results: \n{}".format(results))
    print("Number of dropped solutions: \n " + "GPU: " + format(drop[:, 0]))
    gpu_results = tuple(results[:, 0])
    # cpu_results = tuple(results[:, 1])
    cpu_results = 0
    exp_name = tuple(values)
    utils.save_dict(config, save_folder, 'config_' + parameter_name + '.txt')

    acoc_plotter.plot_bar_graph(gpu_results, cpu_results, exp_name, save=True, show=True, save_folder=SAVE_DIR)

if __name__ == "__main__":
    performance('granularity', [10])
