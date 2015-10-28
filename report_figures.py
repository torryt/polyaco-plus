import acoc_plotter
import numpy as np
from acoc_matrix import AcocMatrix
from matplotlib import pyplot as plt


def plot_points():
    blue = np.array([[1.7, 1.2, .9, .7, 2.3, 1.75, 2.7, 3.3, 3.1],
                     [-0.23, 0.2, 1.05, 1.87, 0.15, 2.08, -.3, 1.1, 2.2],
                     [.0, .0, .0, .0, .0, .0, .0, .0, .0]])
    red = np.array([[2.26, 1.8, 2.3], [1.38, 1.15, 0.8], [1, 1, 1]])
    data = np.concatenate((red, blue), axis=1)

    matrix = AcocMatrix(data, granularity=0.5)
    matrix.plot_matrix(show=False)
    acoc_plotter.plot_data(data, show=False)
    plt.axis('off')
    plt.savefig('points.png', transparent=True)
    plt.savefig('points.svg')
    plt.show()

plot_points()