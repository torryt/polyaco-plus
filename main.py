import acoc
import data_generator as dg
import acoc_plotter as plotter
import utils
import numpy as np

ant_count = 400
iterations = 1
q = 5.0
q_max = 20.0
rho = 0.01
live_plot = True


def run():
    all_ant_scores = np.zeros((iterations, ant_count))
    global_best_polygon = []
    global_best_score = 0

    red = np.insert(dg.uniform_rectangle((1, 3), (2, 4), 200), 2, 0, axis=0)
    blue = np.insert(dg.uniform_rectangle((4, 6), (2, 4), 200), 2, 1, axis=0)
    data = np.concatenate((red, blue), axis=1)

    for i in range(iterations):
        print("\nIteration: {}/{}".format(i + 1, iterations))

        ant_scores, path = \
            acoc.classify(data, ant_count, q, q_max, rho, live_plot)
        utils.print_on_current_line("Best ant score: {}".format(max(ant_scores)))

        all_ant_scores[i, :] = ant_scores
        if max(ant_scores) > global_best_score:
            global_best_polygon = path
            global_best_score = max(ant_scores)

    print("\nGlobal best ant score: {}".format(global_best_score))

    plotter.plot_path_with_data(global_best_polygon, data)
    plotter.plot_ant_scores(ant_scores)


run()
