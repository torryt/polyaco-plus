import pickle
from datetime import datetime
import os.path as osp
import numpy as np
import acoc
from acoc import acoc_plotter
import utils
from acoc.acoc_matrix import AcocMatrix

from config import SAVE_DIR
SAVE = True
SAVE_PHEROMONE_VALUES = False
SHOW_PLOT = True

CONFIG = {
    'ant_count':    500,
    'number_runs':  2,
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
    'gpu':          True
}
data = pickle.load(open('../utils/data_sets.pickle', 'rb'), encoding='latin1')['r_5000']


def run(*args):
    config = dict(CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]
    # data = pickle.load(open('../utils/data_sets.pickle', 'rb'), encoding='latin1')[config['data_set']]
    number_runs = config['number_runs']

    clf = acoc.Classifier(config, SAVE_PHEROMONE_VALUES)
    polygons = []
    diff_x = []

    if number_runs > 1:
        for i in range(number_runs):
            iter_string = "Iteration: {}/{}".format(i + 1, number_runs)
            _, current_best_polygon, _ = clf.classify(data)
            polygons.append(current_best_polygon)

            # if len(polygons) > 1:
            #     diff_x = set(polygons[i-1]) ^ set(polygons[i])
                # diff_y = set(polygons[i]) ^ set(polygons[i-1])
                # diff = set(diff_x).union(set(diff_y))

            utils.print_on_current_line(iter_string)
        return polygons, number_runs, diff_x


def get_all_ant_paths(_data):
    save_folder = generate_folder_name()
    matrix = AcocMatrix(_data, tau_initial=CONFIG.get('tau_init'), granularity=CONFIG.get('granularity'))
    all_polygons, index, diff_poly = run()

    poly1_clean = acoc.remove_twins(all_polygons[0])
    poly2_clean = acoc.remove_twins(all_polygons[1])

    diff = set(poly1_clean) ^ set(poly2_clean)

    # acoc_plotter.plot_multi(all_polygons[1], diff_poly, _data, matrix, SAVE, SHOW_PLOT, save_folder)

    acoc_plotter.plot_path_with_data(poly1_clean, _data, matrix,
                                     save=SAVE, show=SHOW_PLOT, save_folder=save_folder)

    acoc_plotter.plot_path_with_data(poly2_clean, _data, matrix,
                                     save=SAVE, show=SHOW_PLOT, save_folder=save_folder)

    acoc_plotter.plot_path_with_data(diff, _data, matrix,
                                     save=SAVE, show=SHOW_PLOT, save_folder=save_folder, color='r-')

    if SAVE:
        utils.save_dict(CONFIG, save_folder, 'config.txt')
        utils.save_dict(str(all_polygons[index-1]), save_folder, 'path_best.txt')
        utils.save_dict(str(diff_poly), save_folder, 'uncommon.txt')


def generate_folder_name():
    now = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, now) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, now) + '-' + str(iterator)
    return osp.basename(full_path)


if __name__ == "__main__":
    get_all_ant_paths(data)

