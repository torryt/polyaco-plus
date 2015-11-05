import pickle as pick

import numpy as np
from matplotlib import pyplot as plt

from acoc import acoc
from acoc import acoc_plotter
from utils import data_generator as dg
from utils import utils

default_config = {
    'ant_count': 100,
    'iterations': 1,
    'q': 5.0,
    'q_min': 0.1,
    'q_max': 20.0,
    'q_init': 20.0,
    'rho': 0.02,
    'alpha': 1,
    'beta': 0.05,
    'ant_init': 'random',
    'live_plot': False
}

red = dg.uniform_rectangle((1, 4), (2, 5), 500, 0)
blue = dg.uniform_rectangle((5, 8), (2, 5), 500, 1)
data = np.concatenate((red, blue), axis=1)
line_shapes = ['b-', 'g^', 'r-', 'c-', 'm-', 'y-']


def run(*args):
    config = dict(default_config)
    for arg in args:
        config[arg[0]] = arg[1]
    iterations = config['iterations']
    classifier = acoc.Classifier(config)
    all_ant_scores = np.zeros((iterations, config['ant_count']))

    for i in range(iterations):
        utils.print_on_current_line("Iteration: {}/{}".format(i + 1, iterations))
        ant_scores, path = \
            classifier.classify(data, config['live_plot'])
        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0)


def parameter_tester(parameter_name, values):
    plt.clf()
    plots = []
    all_scores = []
    for index, v in enumerate(values):
        print("\nRun {} with value {}".format(index+1, v))
        scores = run((parameter_name, v))
        line = plt.plot(range(len(scores)), scores, line_shapes[index], label=parameter_name + '=' + str(v))
        plots.append(line)
        all_scores.append(scores)

    utils.save_object(all_scores, parameter_name)
    plt.legend()
    plt.axis([0, len(scores), 0, 1])
    acoc_plotter.save_plot()


# parameter_tester('ant_init', ['random', 'weighted', 'static'])
parameter_tester('q_init', [default_config['q_max'], default_config['q_min']])
# parameter_tester('rho', [0.001, 0.01, 0.02, 0.1, 0.3])
# parameter_tester('iterations', [1, 2, 5, 10])