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
    'number_runs':  1,
    'tau_min':      0.001,
    'tau_max':      1.0,
    'tau_init':     0.001,
    'rho':          0.02,
    'alpha':        1,
    'beta':         0.1,
    'ant_init':     'weighted',
    'decay_type':   'probabilistic',
    'data_set':     'rectangle_small',
    'pu_type':      'gpu'
}


def run(*args):
    config = dict(CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]
    data = pickle.load(open('../utils/data_sets.pickle', 'rb'), encoding='latin1')[config['data_set']]
    number_runs = config['number_runs']

    clf = acoc.Classifier(config)
    all_time = np.zeros((number_runs, config['ant_count']))
    all_grid = np.zeros((number_runs, config['ant_count']))
    all_type = np.zeros((number_runs, config['ant_count']))

    for i in range(number_runs):
        iter_string = "Iteration: {}/{}".format(i + 1, number_runs)
        time, path, grid_size, types = \
            clf.classify(data, False, ', ' + iter_string)
        utils.print_on_current_line(iter_string)
        all_time[i, :] = time
        all_grid[i, :] = grid_size
        all_type[i, :] = types

    return all_time.mean(0), all_grid.mean(0), all_type.mean(0)


def performance_tester(pu_name, values, config=CONFIG):
    save_folder = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    save_folder = osp.basename(full_path)
    print("\n\nExperiment for performance '{}' with {}".format(pu_name, values))

    plt.clf()
    all_times = []
    all_types = []
    all_grids = []

    for index, v in enumerate(values):
        print("Run {} with value {}".format(index+1, v))
        time, types, grid = run((pu_name, v))
        all_times.append(time)
        all_types.append(types)
        all_grids.append(grid)
        utils.print_on_current_line('')
    utils.save_dict(config, save_folder, 'config_' + pu_name + '.txt')
    labels = [pu_name + '=' + str(v) for v in values]

    def save_results(data, typ, ylabel, gpu_results, cpu_results, experiments_name):
        utils.save_object(data, save_folder, 'data_' + typ)
        fig = acoc_plotter.plot_bar_graph(gpu_results, cpu_results, experiments_name, save=True, save_folder=SAVE_DIR)

    results = [(all_times, 'time', 'Time'),
               (all_types, 'type', 'Type'),
               (all_grids, 'grid', 'Grid')]
    [save_results(*obj) for obj in results]

if __name__ == "__main__":
    performance_tester('pu_type', ['gpu', 'cpu'])
