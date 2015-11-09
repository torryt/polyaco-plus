import numpy as np
from matplotlib import pyplot as plt

import acoc
from acoc import acoc_plotter
from utils import data_generator as dg
from utils import utils


CONFIG = {
    'ant_count': 500,
    'iterations': 5,
    'q': 5.0,
    'q_min': 0.1,
    'q_max': 20.0,
    'q_init': 0.1,
    'rho': 0.02,
    'alpha': 1,
    'beta': 0.05,
    'ant_init': 'weighted',
    'decay_type': 'random_type',
    'live_plot': False
}

red = dg.uniform_rectangle((1, 4), (2, 5), 500, 0)
blue = dg.uniform_rectangle((5, 8), (2, 5), 500, 1)
data = np.concatenate((red, blue), axis=1)
line_shapes = ['b-', 'g^', 'r-', 'c-', 'm-', 'y-']


def run(*args):
    config = dict(CONFIG)
    for arg in args:
        config[arg[0]] = arg[1]
    iterations = config['iterations']
    clf = acoc.Classifier(config)
    all_ant_scores = np.zeros((iterations, config['ant_count']))

    for i in range(iterations):
        iter_string = "Iteration: {}/{}".format(i + 1, iterations)
        ant_scores, path = \
            clf.classify(data, config['live_plot'], ', ' + iter_string)
        utils.print_on_current_line(iter_string)
        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0)


def parameter_tester(parameter_name, values, config=CONFIG):
    print("\n\nExperiment for parameter '{}' with values {}".format(parameter_name, values))
    plt.clf()
    all_scores = []
    for index, v in enumerate(values):
        print("Run {} with value {}".format(index+1, v))
        scores = run((parameter_name, v))
        all_scores.append(scores)
        utils.print_on_current_line('')

    utils.save_dict(config, 'config_' + parameter_name + '.txt', parameter_name)
    utils.save_object(all_scores, 'data', parameter_name)
    labels = [parameter_name + '=' + str(v) for v in values]
    f1 = acoc_plotter.plot_curves(all_scores, labels)
    acoc_plotter.save_plot(f1, parameter_name)
    f2 = acoc_plotter.plot_smooth_curves(all_scores, labels)
    acoc_plotter.save_plot(f2, parameter_name)

if __name__ == "__main__":
    # parameter_tester('ant_init', ['random', 'static', 'weighted', 'on_global_best', 'chance_of_global_best'])
    parameter_tester('decay_type', ['random_type', 'grad_type'])
    # parameter_tester('q_init', [CONFIG['q_max'], CONFIG['q_min']])
    # parameter_tester('rho', [0.001, 0.01, 0.02, 0.1, 0.3])
    # parameter_tester('iterations', [1, 2, 5, 10])
