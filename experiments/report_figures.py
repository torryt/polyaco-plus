import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from acoc import acoc_plotter


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


def plot_smooth_curves():
    curves = pickle.load(open("save.pickle", "rb"))
    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    values = ['random', 'weighted', 'static']
    for i, c in enumerate(curves):
        y = savgol_filter(c, 51, 2)
        ax1.plot(range(y.shape[0]), y, label=values[i])
        ax2.plot(np.arange(c.shape[0]), c, label=values[i])
    ax1.legend()
    ax2.legend()
    plt.show()


plot_smooth_curves()
#plot_points()
