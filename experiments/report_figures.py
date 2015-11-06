import numpy as np
from matplotlib import pyplot as plt
import pickle
from acoc import acoc_plotter
from acoc.acoc_plotter import plot_smooth_curves, plot_curves


def plot_points():
    blue = np.array([[1.7, 1.2, .9, .7, 2.3, 1.75, 2.7, 3.3, 3.1],
                     [-0.23, 0.2, 1.05, 1.87, 0.15, 2.08, -.3, 1.1, 2.2],
                     [.0, .0, .0, .0, .0, .0, .0, .0, .0]])
    red = np.array([[2.26, 1.8, 2.3], [1.38, 1.15, 0.8], [1, 1, 1]])
    data = np.concatenate((red, blue), axis=1)

    ax = plt.subplot(111)
    acoc_plotter.plot_data(data, ax)
    plt.axis('off')
    acoc_plotter.save_plot()
    plt.savefig('points.svg', bbox_inches='tight')
    # plt.show()

labels = ['random', 'weighted', 'static', 'on_global_best', 'chance_of_global_best']
file_name = '/Users/torrytufteland/Dropbox/ACOC/experiments/Guro/2015-11-06_0156ant_init/data.pickle'
curves = pickle.load(open(file_name, 'rb'), encoding='latin1')
f = plot_curves(curves, labels, loc='lower right')
acoc_plotter.save_plot(f)
# plt.show()
