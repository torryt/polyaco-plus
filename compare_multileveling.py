import numpy as np
from datetime import datetime
import os.path as osp
import pickle

import utils
import acoc
from acoc import acoc_plotter as plotter
from config import SAVE_DIR, CLF_CONFIG

SAVE_FOLDER = 'ML_' + datetime.utcnow().strftime('%Y-%m-%d_%H%M')
full_dir = osp.join(SAVE_DIR, SAVE_FOLDER)

CLF_CONFIG['runs'] = 10
CLF_CONFIG['run_time'] = 700

CLF_CONFIG['max_level'] = 3
CLF_CONFIG['max_level_granularity'] = 17


def run(**kwargs):
    conf = dict(CLF_CONFIG)
    for k, v in kwargs.items():
        conf[k] = v

    clf = acoc.Classifier(conf, SAVE_FOLDER)
    data = pickle.load(open('utils/data_sets.pickle', 'rb'), encoding='latin1')['semicircle_gaussian']

    ant_scores, _, best_ant_history = clf.classify(data)
    print(", Best ant score: {}".format(max(ant_scores)))
    return best_ant_history

with_multi = []
no_multi = []
for i in range(CLF_CONFIG['runs']):
    print("\nRun {}/{}\n".format(i + 1, CLF_CONFIG['runs']))

    with_multi.append(run(multi_level=True))
    no_multi.append(run(multi_level=False, granularity=33))

with_multi_mean = np.array(with_multi).mean(0).tolist()
no_multi_mean = np.array(no_multi).mean(0).tolist()
results = np.array([with_multi_mean, no_multi_mean])



def np_list_to_csv_string(npl):
    return ",".join(list(map(lambda f: "{:.4f}".format(f), npl)))

w_multi_str = np_list_to_csv_string(with_multi_mean)
no_multi_str = np_list_to_csv_string(no_multi_mean)

utils.save_object(results, SAVE_FOLDER, 'results')
plotter.plot_ant_scores(with_multi_mean, save=True, save_folder=SAVE_FOLDER, file_name='with_multi-leveling')
plotter.plot_ant_scores(no_multi_mean, save=True, save_folder=SAVE_FOLDER, file_name='without_multi-leveling')
utils.save_string_to_file(w_multi_str + '\n' + no_multi_str, SAVE_FOLDER, 'results.csv')
utils.save_dict(CLF_CONFIG, SAVE_FOLDER, 'config.json')

print("\nMean best result with multi-leveling: {}".format(w_multi_str))
print("Mean best result without multi-leveling: {}".format(no_multi_str))