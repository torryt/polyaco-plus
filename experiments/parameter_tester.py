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


red = np.insert(dg.uniform_rectangle((1, 4), (2, 5), 500), 2, 0, axis=0)
blue = np.insert(dg.uniform_rectangle((5, 8), (2, 5), 500), 2, 1, axis=0)
data = np.concatenate((red, blue), axis=1)
line_shapes = ['b-', 'g^', 'r-', 'c-', 'm-', 'y-']


def run(ant_count=):
    classifier = acoc.Classifier(_ant_count, q, _q_max, q_min, q_init, rho, _alpha, beta, ant_init)
    all_ant_scores = np.zeros((_iterations, _ant_count))

    for i in range(iterations):
        utils.print_on_current_line("Iteration: {}/{}".format(i + 1, _iterations))
        ant_scores, path = \
            classifier.classify(data, _live_plot)
        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0)


def parameter_tester(parameter_name, values):
    plt.clf()
    plots = []
    all_scores = []
    for index, v in enumerate(values):
        print("\nRun {} with value {}".format(index+1, v))
        scores = run()
        line = plt.plot(range(len(scores)), scores, line_shapes[index], label=parameter_name + '=' + str(v))
        plots.append(line)
        all_scores.append(scores)

    utils.save_object(all_scores, parameter_name)
    plt.legend()
    plt.axis([0, len(scores), 0, 1])
    acoc_plotter.save_plot()


def test_ant_init():
    plt.clf()
    plots = []
    value_tag = 'ant_init'
    values = ['random', 'weighted', 'static']
    all_scores = []
    for index, v in enumerate(values):
        print("\nRun {} with value {}".format(index+1, v))
        scores = run(ant_init=v)
        line = plt.plot(range(len(scores)), scores, line_shapes[index], label=value_tag+'='+str(v))
        plots.append(line)
        all_scores.append(scores)

    utils.save_object(all_scores, value_tag)
    plt.legend()
    plt.axis([0, len(scores), 0, 1])
    acoc_plotter.save_plot()


def test_q_init():
    plt.clf()
    plots = []
    value_tag = 'q_init'
    values = ['random', 'weighted', 'static']
    all_scores = []
    for index, v in enumerate(values):
        print("\nRun {} with value {}".format(index+1, v))
        scores = run(ant_init=v)
        line = plt.plot(range(len(scores)), scores, line_shapes[index], label=value_tag+'='+str(v))
        plots.append(line)
        all_scores.append(scores)

    utils.save_object(all_scores, value_tag)
    plt.legend()
    plt.axis([0, len(scores), 0, 1])
    acoc_plotter.save_plot()


def test_rho():
    plt.clf()
    plots = []
    value_tag = 'rho'
    values = [0.001, 0.01, 0.02, 0.1, 0.3]
    print("Testing parameter " + value_tag)
    for index, v in enumerate(values):
        print("\nRun {} with value {}".format(index+1, v))
        scores = run(rho=v)
        line = plt.plot(range(len(scores)), scores, line_shapes[index], label=value_tag+'='+str(v))
        plots.append(line)

    plt.legend()
    plt.axis([0, len(scores), 0, 1])
    acoc_plotter.save_plot()


def test_iterations():
    plt.clf()
    plots = []
    value_tag = 'iterations'
    values = [1, 2, 5, 10]
    print("Testing parameter " + value_tag)
    for index, v in enumerate(values):
        print("\nRun {} with value {}".format(index+1, v))
        scores = run(iterations=v)
        line = plt.plot(range(len(scores)), scores, line_shapes[index], label=value_tag+'='+str(v))
        plots.append(line)

    plt.legend()
    plt.axis([0, len(scores), 0, 1])
    acoc_plotter.save_plot()

test_q_init()
