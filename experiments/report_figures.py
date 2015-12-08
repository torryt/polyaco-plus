import pickle
import numpy as np
import os.path as osp
from uuid import uuid4
from matplotlib import pyplot as plt

from acoc import acoc_plotter as ac
from acoc.acoc_plotter import plot_smooth_curves, plot_curves


def plot_points():
    blue = np.array([[1.7, 1.2, .9, .7, 2.3, 1.75, 2.7, 3.3, 3.1],
                     [-0.23, 0.2, 1.05, 1.87, 0.15, 2.08, -.3, 1.1, 2.2],
                     [.0, .0, .0, .0, .0, .0, .0, .0, .0]])
    red = np.array([[2.26, 1.8, 2.3], [1.38, 1.15, 0.8], [1, 1, 1]])
    data = np.concatenate((red, blue), axis=1)

    ax = plt.subplot(111)
    ac.plot_data(data, ax)
    plt.axis('off')
    ac.save_plot()
    plt.savefig('points.svg', bbox_inches='tight')
    # plt.show()


def plot_curves_from_data(file_name):
    # labels = ['random', 'weighted', 'static', 'on_global_best', 'chance_of_global_best']
    # labels = ['probabilistic', 'gradual']
    labels = ['0.01', '0.1', '1.0', '10.0']
    curves = pickle.load(open(file_name, 'rb'), encoding='latin1')

    f1 = plot_curves(curves, labels, loc='upper left')
    f2 = plot_smooth_curves(curves, labels, loc='upper left')

    base_path = osp.dirname(file_name)
    f1.savefig(osp.join(base_path, str(uuid4()) + '.eps'))
    f1.savefig(osp.join(base_path, str(uuid4()) + '.png'))
    f2.savefig(osp.join(base_path, str(uuid4()) + '.eps'))
    f2.savefig(osp.join(base_path, str(uuid4()) + '.png'))
    # ac.save_plot(f)
    # plt.show()


def plot_all_data_sets():
    data_sets = pickle.load(open('../data_sets.pickle', 'rb'), encoding='latin1')
    for key in data_sets:
        data = data_sets[key]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ac.hide_top_and_right_axis(ax)
        # plt.axis('off')
        ac.plot_data(data, ax)
        ac.save_plot(fig)

if __name__ == "__main__":
    # plot_all_data_sets()
    plot_curves_from_data('/Users/torrytufteland/Dropbox/ACOC/experiments/p_q kort/data.pickle')