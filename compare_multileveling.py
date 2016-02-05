import numpy as np
from datetime import datetime
import os.path as osp

from acoc import acoc_plotter as plotter
from acoc_runner import run, clf_config
from config import SAVE_DIR

SAVE_FOLDER = 'ML_' + datetime.utcnow().strftime('%Y-%m-%d_%H%M')

full_dir = osp.join(SAVE_DIR, SAVE_FOLDER)
runs = 3

with_multi = np.empty((runs, clf_config['run_time']))
no_multi = np.empty((runs, clf_config['run_time']))
for i in range(runs):
    print("\nRun {}/{}\n".format(i + 1, runs))

    with_multi[i] = run(multi_level=True)
    no_multi[i] = run(multi_level=False, granularity=17)

np.set_printoptions(precision=3)
plotter.plot_ant_scores(with_multi.mean(0), save=True, save_folder=SAVE_FOLDER)
plotter.plot_ant_scores(no_multi.mean(0), save=True, save_folder=SAVE_FOLDER)

with open(osp.join(full_dir, "with_multileveling.txt"), "w") as text_file:
    print(with_multi.mean(0), file=text_file)

with open(osp.join(full_dir, "without_multileveling.txt"), "w") as text_file:
    print(no_multi.mean(0), file=text_file)

print("\nMean best result with multi-leveling: {}".format(with_multi.mean(0)))
print("Mean best result without multi-leveling: {}".format(no_multi.mean(0)))
