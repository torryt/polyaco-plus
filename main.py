import acoc
import data_generator as dg
import acoc_plotter as plotter
import utils
import numpy as np
from is_point_inside import is_point_inside

ant_count = 200
iterations = 2
q = 5.0
q_max = 20.0
rho = 0.01
alpha = 1
beta = 0.01
live_plot = True

classifier = acoc.Classifier(ant_count, q, q_max, rho, alpha, beta)


def run():
    all_ant_scores = np.zeros((iterations, ant_count))
    global_best_polygon = []
    global_best_score = 0

    red = np.insert(dg.uniform_rectangle((1, 3), (2, 4), 500), 2, 0, axis=0)
    blue = np.insert(dg.uniform_rectangle((4, 6), (2, 4), 500), 2, 1, axis=0)
    # red = dg.uniform_circle(3.0, 500, 1)
    # blue = dg.uniform_circle(2.0, 500, 0)
    data = np.concatenate((red, blue), axis=1)

    for i in range(iterations):
        print("\nIteration: {}/{}".format(i + 1, iterations))

        ant_scores, path = \
            classifier.classify(data, live_plot)
        utils.print_on_current_line("Best ant score: {}".format(max(ant_scores)))

        all_ant_scores[i, :] = ant_scores
        if max(ant_scores) > global_best_score:
            global_best_polygon = path
            global_best_score = max(ant_scores)

    true_positives = 0
    false_positives = 0
    points = data.T.tolist()
    for p in points:
        if is_point_inside(p, global_best_polygon):
            if p[2] == 0:
                true_positives += 1
            else:
                false_positives += 1

    total_red = len([p for p in points if p[2] == 0])
    total_blue = len([p for p in points if p[2] == 0])
    print("\n\nTrue positives: {}/{}\nFalse positives: {}/{}".format(true_positives, total_red,
                                                                   false_positives, total_blue))
    print("Global best ant score: {}".format(global_best_score))

    plotter.plot_path_with_data(global_best_polygon, data)
    plotter.plot_ant_scores(ant_scores)


run()
