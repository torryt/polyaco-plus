import numpy as np
from matplotlib import pyplot as plt

from acoc import acoc_plotter as ap
from utils.data_generator import uniform_circle


def main():
    red = uniform_circle(1.0, 500, 0, spread=0.4)
    blue = uniform_circle(2.0, 500, 1, spread=0.4)
    data = np.concatenate((red, blue), axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ap.hide_top_and_right_axis(ax)
    # plt.axis('off')
    ap.plot_data(data, ax)
    ap.save_plot(fig)

main()