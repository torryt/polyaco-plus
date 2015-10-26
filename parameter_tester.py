import acoc
import data_generator as dg
import acoc_plotter as plotter
from matplotlib import pyplot as plt
import utils
import numpy as np

ant_count = 400
iterations = 1
q = 5.0
q_max = 20.0
rho = 0.01
alpha = 1
# beta = 0.1
live_plot = False

red = np.insert(dg.uniform_rectangle((1, 4), (2, 5), 500), 2, 0, axis=0)
blue = np.insert(dg.uniform_rectangle((5, 8), (2, 5), 500), 2, 1, axis=0)
data = np.concatenate((red, blue), axis=1)


def run(beta):
    classifier = acoc.Classifier(ant_count, q, q_max, rho, alpha, beta)
    all_ant_scores = np.zeros((iterations, ant_count))

    for i in range(iterations):
        print("\nIteration: {}/{}".format(i + 1, iterations))

        ant_scores, path = \
            classifier.classify(data, live_plot)

        all_ant_scores[i, :] = ant_scores

    return all_ant_scores.mean(0)

plots = []
values = [0.01, 0.05, 0.1, 0.2, 0.4]
line_shapes = ['b-', 'g-', 'r-', 'c-', 'm-']
for index, v in enumerate(values):
    scores = run(beta=v)
    line = plt.plot(range(len(scores)), scores, line_shapes[index], label='beta='+str(v))
    plots.append(line)

plt.legend()
plt.axis([0, len(scores), 0, 1])

plt.show()

