import pickle
import time

import numpy as np

import acoc
import utils
from utils import generate_folder_name

CONFIG = {
    'ant_count':    1500,
    'number_runs':  20,
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


def benchmark_cost_function(data_sets):
    polygon = pickle.load(open('../utils/good_path_for_rectangle.pickle', 'rb'))

    save_folder = generate_folder_name()
    iterations = 100
    results = np.empty((len(data_sets), iterations, 2), dtype=float)

    for i, data_set_name in enumerate(data_sets):
        data = pickle.load(open('../utils/data_sets.pickle', 'rb'))[data_set_name]

        print("\nRun {} with value {}".format(i+1, data_set_name))
        for j in range(iterations):
            utils.print_on_current_line('Iteration {}/{}'.format(j, iterations))
            start_gpu = time.clock()
            acoc.cost_function_gpu(data.T, polygon)
            end_gpu = time.clock()
            results[i][j][0] = end_gpu - start_gpu

            start_cpu = time.clock()
            acoc.cost_function_cpu(data.T, polygon)
            end_cpu = time.clock()
            results[i][j][1] = end_cpu - start_cpu

    mean_results = np.mean(results, axis=1)
    print("Results: \n{}".format(mean_results))
    utils.save_object(mean_results, save_folder, 'results')


if __name__ == "__main__":
    # benchmark('granularity', [10, 20, 30, 40, 50, 100])
    # benchmark('data_set', ['r_50', 'r_500', 'r_5000', 'r_50000', 'r_500000'])
    # benchmark('data_set', ['r_50', 'r_500', 'r_5000'])
    benchmark_cost_function(['r_50', 'r_500', 'r_5000', 'r_50000'])
