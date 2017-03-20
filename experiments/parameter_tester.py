import os.path as osp
from copy import copy
from sklearn.cross_validation import train_test_split
import sys
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import acoc
import utils

from utils import data_manager, generate_folder_name
from config import SAVE_DIR, CLASSIFIER_CONFIG

CLASSIFIER_CONFIG.runs = 8
CLASSIFIER_CONFIG.level_convergence_rate = 200
CLASSIFIER_CONFIG.max_level = 3
CLASSIFIER_CONFIG.data_set = 'iris'

CLASSIFIER_CONFIG.training_test_split = True


def run(*args):
    config = copy(CLASSIFIER_CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]

    data_set = data_manager.load_data_set(config.data_set)
    classifier_scores = []
    for nrun in range(config.runs):
        X = data_set.data
        y = data_set.target
        class_indices = list(set(y))

        if config.training_test_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33)
        else:
            X_train = X_test = X
            y_train = y_test = y

        clf = acoc.PolyACO(X_train.shape[1], class_indices, config)
        clf.train(X_train, y_train)
        predictions = clf.evaluate(X_test)

        classifier_scores.append(acoc.compute_score(predictions, y_test))
        print("\nRun {}/{} score: {:.4f}".format(nrun + 1, config.runs, classifier_scores[-1]))
    return classifier_scores


def parameter_tester(parameter_name, values, save_folder=None):
    if save_folder is None:
        save_folder = utils.generate_folder_name()
    print("\n\nExperiment for parameter '{}' with values {}".format(parameter_name, values))

    plt.clf()
    all_scores = []
    for index, v in enumerate(values):
        print("Run {} with value {}".format(index + 1, v))
        scores = run((parameter_name, v))
        all_scores.append(scores)
        utils.print_on_current_line('')
    header = ','.join(str(s) for s in values)
    result_str = header + '\n' + ','.join(["{:.4f}".format(sum(s) / CLASSIFIER_CONFIG.runs) for s in all_scores]) + \
                 '\n\n' + 'all scores:\n'

    for a in all_scores:
        result_str += ','.join('{:.4f}'.format(s) for s in a) + '\n'
    utils.save_string_to_file(result_str, parent_folder=save_folder, file_name='result_' + parameter_name + '.txt')


if __name__ == "__main__":
    folder = generate_folder_name('tuning-'.format(CLASSIFIER_CONFIG.data_set))
    utils.save_dict(CLASSIFIER_CONFIG, folder, file_name='base_config.txt')
    parameter_tester('rho', [0.02, 0.05, 0.07], save_folder=folder)
    parameter_tester('beta', [0.02, 0.05, 0.07], save_folder=folder)
