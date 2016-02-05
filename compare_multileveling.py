import numpy as np
from datetime import datetime
import os.path as osp

from acoc import acoc_plotter as plotter
from main import run, clf_config
from config import SAVE_DIR

SAVE_FOLDER = 'ML_' + datetime.utcnow().strftime('%Y-%m-%d_%H%M')

full_dir = osp.join(SAVE_DIR, SAVE_FOLDER)
runs = 5

with_multi = []
no_multi = []
for i in range(runs):
    print("\nRun {}/{}\n".format(i + 1, runs))

    with_multi.append(run(multi_level=True))
    no_multi.append(run(multi_level=False, granularity=33))

with_multi_mean = np.array(with_multi).mean(0).tolist()
no_multi_mean = np.array(no_multi).mean(0).tolist()


def np_list_to_csv_string(npl):
    return ",".join(list(map(lambda f: "{:.4f}".format(f), npl)))

w_multi_str = np_list_to_csv_string(with_multi_mean)
no_multi_str = np_list_to_csv_string(no_multi_mean)

plotter.plot_ant_scores(with_multi_mean, save=True, save_folder=SAVE_FOLDER, file_name='with_multi-leveling')
plotter.plot_ant_scores(no_multi_mean, save=True, save_folder=SAVE_FOLDER, file_name='without_multi-leveling')

with open(osp.join(full_dir, "results.csv"), "w") as text_file:
    print(w_multi_str + '\n' + no_multi_str, file=text_file)

print("\nMean best result with multi-leveling: {}".format(w_multi_str))
print("Mean best result without multi-leveling: {}".format(no_multi_str))