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

CLASSIFIER_CONFIG['runs'] = 20
CLASSIFIER_CONFIG['run_time'] = 100
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

with_multi = []
high_gran = []
low_gran = []
for i in range(CLASSIFIER_CONFIG['runs']):
    print("\nRun {}/{}\n".format(i + 1, CLASSIFIER_CONFIG['runs']))

    with_multi.append(run(multi_level=True))
    high_gran.append(run(multi_level=False, granularity=33))
    low_gran.append(run(multi_level=False, granularity=3))


def np_list_to_csv_string(npl):
    return ",".join(list(map(lambda f: "{:.4f}".format(f), npl)))

results = []
csv = []
for arr in [with_multi, high_gran, low_gran]:
    l = np.array(arr).mean(0).tolist()
    csv.append(np_list_to_csv_string(l))
    results.append(l)


utils.save_object(results, SAVE_FOLDER, 'results')
utils.save_string_to_file("\n".join(csv), SAVE_FOLDER, 'results.csv')
utils.save_dict(CLASSIFIER_CONFIG, SAVE_FOLDER, 'config.json')

data = np.array(results)
x = range(data.shape[1])
labels = ['With multi-leveling', 'High granularity', 'Low granularity']
fig, ax = plt.subplots()

plotter.hide_top_and_right_axis(ax)
ax.yaxis.grid(color='gray')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Best polygon solution')

l1, = ax.plot(x, data[0], plotter.COLORS[0], label=labels[0])
l2, = ax.plot(x, data[1], plotter.COLORS[1], label=labels[1])
l3, = ax.plot(x, data[2], plotter.COLORS[2], label=labels[2])


plt.legend(labels, loc='lower right')
plotter.save_plot(fig, SAVE_FOLDER, 'results')

# plt.show()
#fig1.savefig('fig1.eps')

print("\nMean best result with multi-leveling: {}".format(csv[0]))
print("Mean best result without multi-leveling: {}".format(csv[1]))