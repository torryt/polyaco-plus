import numpy as np
import pickle
from datetime import datetime
import os.path as osp
import time

import acoc
import utils

from config import SAVE_DIR

CONFIG = {
    'ant_count':    1500,
    'number_runs':  2,
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
    data = pickle.load(open('../utils/data_sets.pickle', 'rb'))[config['data_set']]
    number_runs = config['number_runs']

    clf = acoc.Classifier(config)

    run_times = np.zeros(number_runs, dtype=float)
    for i in range(number_runs):
        iter_string = "Iteration: {}/{}".format(i + 1, number_runs)

        start = time.clock()
        clf.classify(data, print_string=', ' + iter_string)
        end = time.clock()
        run_times[i] = end - start

        utils.print_on_current_line(iter_string)
    return np.mean(run_times)


def generate_folder_name():
    now = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, now) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, now) + '-' + str(iterator)
    return osp.basename(full_path)


def benchmark(parameter_name, values, config=CONFIG):
    save_folder = generate_folder_name()
    print("\n\nBenchmark for parameter '{}' with values {}".format(parameter_name, values))

    results = np.empty((len(values), 2), dtype=float)
    for index, v in enumerate(values):
        print("Run {} with value {} on GPU".format(index+1, v))
        results[index, 0] = run((parameter_name, v), ('gpu', True))
        utils.clear_current_line()

        print("Run {} with value {} on CPU".format(index+1, v))
        results[index, 1] = run((parameter_name, v), ('gpu', False))
        utils.clear_current_line()

    result_str = "Results: \n{}".format(results)
    print(result_str)
    utils.save_dict(config, save_folder, 'config_' + parameter_name + '.txt')
    utils.save_string_to_file(result_str, save_folder, 'results.txt')


def benchmark_cost_function(parameter_name, values, config=CONFIG):
    data = pickle.load(open('utils/data_sets.pickle', 'rb'))[config['data_set']]
    # TODO: Fill in
    polygon = []

    save_folder = generate_folder_name()
    results = np.empty((len(values), 2), dtype=float)

    for index, v in enumerate(values):
        print("Run {} with value {} on GPU".format(index+1, v))
        results[index, 0] = acoc.cost_function_gpu(polygon, data)
        utils.clear_current_line()

        print("Run {} with value {} on CPU".format(index+1, v))
        results[index, 1] = acoc.cost_function_cpu(polygon, data)
        utils.clear_current_line()
    print("Results: \n{}".format(results))
    utils.save_object(results, save_folder, 'results')


if __name__ == "__main__":
    # benchmark('granularity', [10, 20, 30, 40, 50, 100])
    # benchmark('data_set', ['r_50', 'r_500', 'r_5000', 'r_50000', 'r_500000'])
    benchmark('data_set', ['r_500'])
    # benchmark_cost_function('granularity', [10, 20, 40, 100])
