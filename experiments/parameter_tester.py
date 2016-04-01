from datetime import datetime
import os.path as osp
from copy import copy
from sklearn.cross_validation import train_test_split
import sys
from matplotlib import pyplot as plt
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import acoc
import utils
from utils import data_manager
from config import SAVE_DIR, CLASSIFIER_CONFIG

NUMBER_RUNS = 50
DATA_SET = 'iris'


def run(*args):
    config = copy(CLASSIFIER_CONFIG)
    for conf in args:
        config[conf[0]] = conf[1]

    data_set = data_manager.load_data_set(DATA_SET)
    classifier_scores = []
    for nrun in range(NUMBER_RUNS):
        X = data_set.data
        y = data_set.target
        class_indices = list(set(y))

        # Split data into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)

        clf = acoc.PolyACO(X.shape[1], class_indices, config)
        clf.train(X_train, y_train, ', Run: {}/{}'.format(nrun, NUMBER_RUNS))
        predictions = clf.evaluate(X_test)

        classifier_scores.append(acoc.compute_score(predictions, y_test))
    return sum(classifier_scores) / NUMBER_RUNS


def parameter_tester(parameter_name, values, config=CLASSIFIER_CONFIG):
    save_folder = datetime.utcnow().strftime('%Y-%m-%d_%H%M')
    iterator = 0
    full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    while osp.exists(full_path):
        iterator += 1
        full_path = osp.join(SAVE_DIR, save_folder) + '-' + str(iterator)
    save_folder = osp.basename(full_path)
    print("\n\nExperiment for parameter '{}' with values {}".format(parameter_name, values))

    plt.clf()
    scores = []
    for index, v in enumerate(values):
        print("Run {} with value {}".format(index+1, v))
        score = run((parameter_name, v))
        scores.append(score)
        utils.print_on_current_line('')
    utils.save_dict(config, save_folder, 'config_' + parameter_name + '.txt')
    header = ','.join(str(s) for s in values)
    result_str = header + '\n' + ','.join([str(s) for s in scores])
    utils.save_string_to_file(result_str, parent_folder=save_folder, file_name='result.txt')


if __name__ == "__main__":
    parameter_tester('one_less_class', [True, False])
