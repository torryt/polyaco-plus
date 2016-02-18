import numpy as np
from datetime import datetime
import os.path as osp
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import utils
import acoc
from acoc import acoc_plotter as plotter
from config import SAVE_DIR, CLASSIFIER_CONFIG

SAVE_FOLDER = 'ML_' + datetime.utcnow().strftime('%Y-%m-%d_%H%M')
full_dir = osp.join(SAVE_DIR, SAVE_FOLDER)

CLASSIFIER_CONFIG['runs'] = 1
CLASSIFIER_CONFIG['run_time'] = 5
CLASSIFIER_CONFIG['max_level'] = 4
CLASSIFIER_CONFIG['max_level_granularity'] = 33


def run(**kwargs):
    conf = dict(CLASSIFIER_CONFIG)
    for k, v in kwargs.items():
        conf[k] = v

    clf = acoc.Classifier(conf, SAVE_FOLDER)
    data = pickle.load(open('utils/data_sets.pickle', 'rb'), encoding='latin1')['semicircle_gaussian']

    best_ant_history, _ = clf.classify(data)
    print(", Best ant score: {}".format(max(best_ant_history)))
    return best_ant_history


configurations = [
    {'label': 'With multi-leveling', 'multi_level': True},
    {'label': r'$\mu = 3$', 'multi_level': False, 'granularity': 3},
    {'label': r'$\mu = 9$', 'multi_level': False, 'granularity': 9},
    {'label': r'$\mu = 33$', 'multi_level': False, 'granularity': 33}
]

results = [[] for i in range(len(configurations))]
for i in range(CLASSIFIER_CONFIG['runs']):
    print("\nRun {}/{}\n".format(i + 1, CLASSIFIER_CONFIG['runs']))
    for j, c in enumerate(configurations):
        results[j].append(run(**c))

mean_results = np.array(results).mean(1).tolist()


def np_list_to_csv_string(npl):
    return ",".join(list(map(lambda f: "{:.4f}".format(f), npl)))

csv = []
for arr in mean_results:
    csv.append(np_list_to_csv_string(arr))

utils.save_object(mean_results, SAVE_FOLDER, 'results')
utils.save_string_to_file("\n".join(csv), SAVE_FOLDER, 'results.csv')
utils.save_dict(CLASSIFIER_CONFIG, SAVE_FOLDER, 'config.json')


data = np.array(mean_results)
x = range(data.shape[1])
labels = [c['label'] for c in configurations]
fig, ax = plt.subplots()

plotter.hide_top_and_right_axis(ax)
ax.yaxis.grid(color='gray')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Best polygon solution')

lines = []
for i in range(len(configurations)):
    lines.append(ax.plot(x, data[i], plotter.COLORS[i], label=labels[i]))

plt.legend(labels, loc='lower right')
plotter.save_plot(fig, SAVE_FOLDER, 'results')

# plt.show()
#fig1.savefig('fig1.eps')

print("\nMean best result with multi-leveling: {}".format(csv[0]))
print("Mean best result without multi-leveling: {}".format(csv[1]))