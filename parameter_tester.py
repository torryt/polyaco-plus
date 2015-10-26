import acoc
import data_generator as dg
import acoc_plotter as plotter
from matplotlib import pyplot as plt
import utils
import numpy as np
from time import gmtime, strftime
import os

ant_count = 10
iterations = 1
q = 5.0
q_min = 0.1
q_max = 20.0
rho = 0.02
alpha = 1
beta = 0.1
live_plot = False

red = np.insert(dg.uniform_rectangle((1, 4), (2, 5), 500), 2, 0, axis=0)
blue = np.insert(dg.uniform_rectangle((5, 8), (2, 5), 500), 2, 1, axis=0)
data = np.concatenate((red, blue), axis=1)


def run(new_rho=rho, new_beta=beta):
    classifier = acoc.Classifier(ant_count, q, q_max, q_min, new_rho, alpha, new_beta)
    all_ant_scores = np.zeros((iterations, ant_count))

    for i in range(iterations):
        utils.print_on_current_line("Iteration: {}/{}".format(i + 1, iterations))

        ant_scores, path = \
            classifier.classify(data, live_plot)

        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0)

plots = []
value_tag = 'beta'
values = [0.005, 0.01, 0.02, 0.04]
line_shapes = ['b-', 'g-', 'r-', 'c-']
for index, v in enumerate(values):
    print("\nRun {} with value {}".format(index+1, v))
    scores = run(new_beta=v)
    line = plt.plot(range(len(scores)), scores, line_shapes[index], label=value_tag+'='+str(v))
    plots.append(line)

plt.legend()
plt.axis([0, len(scores), 0, 1])

directory = 'experiments/' + strftime("%Y-%m-%d %H%M%S/", gmtime())
if not os.path.exists(directory):
    os.makedirs(directory)
    plt.savefig(os.path.join(directory, 'result.png'))
    plt.savefig(os.path.join(directory, 'result.svg'))
else:
    print("Could not create directory. Directory '{}' already exists".format(directory))