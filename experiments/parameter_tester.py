from datetime import datetime
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

CLASSIFIER_CONFIG.runs = 100
CLASSIFIER_CONFIG.max_level = 5
CLASSIFIER_CONFIG.data_set = 'breast_cancer'


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

        # Split data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)

        clf = acoc.PolyACO(X.shape[1], class_indices, config)
        clf.train(X_train, y_train, ', Run: {}/{}'.format(nrun + 1, config.runs))
        predictions = clf.evaluate(X_test)

        classifier_scores.append(acoc.compute_score(predictions, y_test))
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


def experiment(title):
    save_folder = generate_folder_name(title)
    utils.save_dict(CLASSIFIER_CONFIG, save_folder, 'base_config.txt')
    parameter_tester('level_convergence_rate', [20, 100, 200, 800, 1600], save_folder)


if __name__ == "__main__":
    # CLASSIFIER_CONFIG.runs = 20
    # experiment('tuning_breast_cancer')

    CLASSIFIER_CONFIG.runs = 100
    CLASSIFIER_CONFIG.max_level = 5
    CLASSIFIER_CONFIG.level_convergence_rate = 800

    folder = generate_folder_name('iris-tuned')
    utils.save_dict(CLASSIFIER_CONFIG, folder, file_name='base_config.txt')
    parameter_tester('data_set', ['iris'], save_folder=folder)
