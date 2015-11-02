import acoc
import data_generator as dg
import acoc_plotter
from matplotlib import pyplot as plt
import utils
import numpy as np

ant_count = 200
iterations = 5
q = 5.0
q_min = 0.1
q_max = 20.0
q_init = q_max
rho = 0.02
alpha = 1
beta = 0.05
live_plot = False

red = np.insert(dg.uniform_rectangle((1, 4), (2, 5), 500), 2, 0, axis=0)
blue = np.insert(dg.uniform_rectangle((5, 8), (2, 5), 500), 2, 1, axis=0)
data = np.concatenate((red, blue), axis=1)


def run(new_rho=rho, new_beta=beta, new_q=q, new_q_min=q_min, new_iter=iterations, new_q_init=q_init):
    classifier = acoc.Classifier(ant_count, new_q, q_max, new_q_min, new_q_init, new_rho, alpha, new_beta)
    all_ant_scores = np.zeros((iterations, ant_count))

    for i in range(new_iter):
        utils.print_on_current_line("Iteration: {}/{}".format(i + 1, iterations))

        ant_scores, path = \
            classifier.classify(data, live_plot)

        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0)

plots = []
value_tag = 'rho'
values = [0.001, 0.01, 0.02, 0.1, 0.3]
line_shapes = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-']
for index, v in enumerate(values):
    print("\nRun {} with value {}".format(index+1, v))
    scores = run(new_rho=v)
    line = plt.plot(range(len(scores)), scores, line_shapes[index], label=value_tag+'='+str(v))
    plots.append(line)

plt.legend()
plt.axis([0, len(scores), 0, 1])
acoc_plotter.save_plot()
