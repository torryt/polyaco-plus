import numpy as np
from datetime import datetime
import os.path as osp
import os
import pickle
import json

import acoc
from acoc import acoc_plotter as plotter
from config import SAVE_DIR, CLF_CONFIG

SAVE_FOLDER = 'ML_' + datetime.utcnow().strftime('%Y-%m-%d_%H%M')
full_dir = osp.join(SAVE_DIR, SAVE_FOLDER)

CLF_CONFIG['runs'] = 2
CLF_CONFIG['run_time'] = 3


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


def np_list_to_csv_string(npl):
    return ",".join(list(map(lambda f: "{:.4f}".format(f), npl)))

w_multi_str = np_list_to_csv_string(with_multi_mean)
no_multi_str = np_list_to_csv_string(no_multi_mean)

if not osp.exists(full_dir):
    os.makedirs(full_dir)

plotter.plot_ant_scores(with_multi_mean, save=True, save_folder=SAVE_FOLDER, file_name='with_multi-leveling')
plotter.plot_ant_scores(no_multi_mean, save=True, save_folder=SAVE_FOLDER, file_name='without_multi-leveling')

with open(osp.join(full_dir, "results.csv"), "w") as text_file:
    print(w_multi_str + '\n' + no_multi_str, file=text_file)
with open(osp.join(full_dir, "config.json"), "w") as json_file:
    json_file.write(json.dumps(CLF_CONFIG, sort_keys=True, indent=2, separators=(',', ': ')))


print("\nMean best result with multi-leveling: {}".format(w_multi_str))
print("Mean best result without multi-leveling: {}".format(no_multi_str))