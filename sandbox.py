import numpy as np
from matplotlib import pyplot as plt

from acoc import acoc_plotter as ap
from utils.data_generator import gaussian_circle
from utils import data_generator
from main import run


def main():
    red = gaussian_circle(1.0, 500, 0, spread=0.4)
    blue = gaussian_circle(2.0, 500, 1, spread=0.4)
    data = np.concatenate((red, blue), axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ap.hide_top_and_right_axis(ax)
    # plt.axis('off')
    ap.plot_data(data, ax)
    ap.save_plot(fig)


def generate_data():
    data_generator.generate_various_sized_rectangles([50, 500, 5000, 50000, 500000])

if __name__ == "__main__":
    generate_data()
    # main()
