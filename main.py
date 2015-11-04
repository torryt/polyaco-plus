import acoc
import data_generator as dg
import acoc_plotter as plotter
import utils
import numpy as np

ant_count = 200
iterations = 1
q = 0.1
q_min = 0.1
q_max = 10.0
q_init = q_max
rho = 0.01
alpha = 1
beta = 0.05
live_plot = True
save = True
show_plot = True
ant_init = 'random'

classifier = acoc.Classifier(ant_count, q, q_min, q_max, q_init, rho, alpha, beta, ant_init)


def run():
    all_ant_scores = np.zeros((iterations, ant_count))
    global_best_polygon = []
    global_best_score = 0

    # red = dg.uniform_circle(3.0, 500, 1)
    # blue = dg.uniform_circle(2.0, 500, 0)
    red = dg.uniform_rectangle((1, 3), (2, 4), 500, 0)
    blue = dg.uniform_rectangle((4, 6), (2, 4), 500, 1)
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

    score = acoc.polygon_score(global_best_polygon, data)
    if save:
        plotter.save_object(all_ant_scores.mean(0))
    print("\n\nGlobal best score(points) {}".format(score))
    print("Global best score(|solution| and points): {}".format(global_best_score))

    plotter.plot_path_with_data(global_best_polygon, data, save=save, show=show_plot)
    plotter.plot_ant_scores(all_ant_scores.mean(0), save=save, show=show_plot)


run()
