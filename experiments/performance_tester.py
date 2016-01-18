import numpy as np
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
import os.path as osp

import acoc
from acoc import acoc_plotter
import utils
from timeit import timeit
import time

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
    'data_set':     'rectangle_small',
    'gpu':          True,
    'granularity':  10
}


def run(*args):
    config = dict(CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]
    data = pickle.load(open('../utils/data_sets.pickle', 'rb'), encoding='latin1')[config['data_set']]
    number_runs = config['number_runs']
    run_time = np.zeros((number_runs, config['ant_count']))

    for i in range(number_runs):
        start = time.clock()
        clf = acoc.Classifier(config)
        clf.classify(data, False)
        end = time.clock()
        time_spent = end - start
        run_time[i, :] = time_spent

    return run_time


def performance_tester(parameter_name, values, config=CONFIG):
    save_folder = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    save_folder = osp.basename(full_path)
    print("\n\nExperiment for performance '{}' with {}".format(parameter_name, values))

    plt.clf()
    all_times = []
    all_grids = []

    for index, v in enumerate(values):
        print("Run {} with value {}".format(index+1, v))
        time_spent_gpu = run((parameter_name, v))
        all_times.append(time_spent_gpu)
        utils.print_on_current_line('')

    utils.save_dict(config, save_folder, 'config_' + parameter_name + '.txt')
    labels = [parameter_name + '=' + str(v) for v in values]

    def save_results(data, typ, gpu_results, cpu_results, granularity):
        utils.save_object(data, save_folder, 'data_' + typ)
        fig = acoc_plotter.plot_bar_graph(gpu_results, cpu_results, granularity, save=True, save_folder=SAVE_DIR)

    results = [(all_times, 'time', 'Time'),
               (all_types, 'type', 'Type'),
               (all_grids, 'grid', 'Grid')]
    [save_results(*obj) for obj in results]

if __name__ == "__main__":
    performance_tester('granularity', [10, 20])
