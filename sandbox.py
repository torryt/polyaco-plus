import numpy as np
from matplotlib import pyplot as plt

from acoc import acoc_plotter as ap
from utils.data_generator import gaussian_circle
from utils import data_generator
import acoc
from acoc_runner import run
import acoc.acoc_plotter as plotter
from utils import data_generator as dg


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


def test_run():
    clf_config = {
        'ant_count':    100,
        'tau_min':      0.001,
        'tau_max':      1.0,
        'tau_init':     0.001,
        'rho':          0.02,
        'alpha':        1,
        'beta':         0.01,
        'ant_init':     'weighted',
        'decay_type':   'probabilistic',
        'gpu':          True,
        'granularity':  10
    }
    clf = acoc.Classifier(clf_config)
    data = dg.generate_rectangle_set(500)
    _, polygon, _ = clf.classify(data)
    print("\nPolygon length: " + str(len(polygon)))
    twins_removed = acoc.remove_twins(polygon)
    print("Twins removed length: " + str(len(acoc.remove_twins(polygon))))
    print("Set length: " + str(len(set(polygon))))
    plotter.plot_path_with_data(polygon, data, clf.matrix, show=True)
    plotter.plot_path_with_data(acoc.remove_twins(polygon), data, clf.matrix, show=True)



if __name__ == "__main__":
    # generate_data()
    # main()
    test_run()
