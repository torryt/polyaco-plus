import numpy as np
from matplotlib import pyplot as plt
import pickle
from acoc import acoc_plotter
from acoc.acoc_plotter import plot_smooth_curves


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

labels = ['q_init=20.0', 'q_init=0.1']
curves = pickle.load(open('/Users/torrytufteland/Dropbox/ACOC/experiments/q_init/q_init.pickle', 'rb'))
f = plot_smooth_curves(curves, labels)
acoc_plotter.save_plot(f)
#plot_points()
