from __future__ import division
import os
from datetime import datetime
import numpy as np
# import matplotlib
# matplotlib.use('Agg')

from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from utils import generate_folder_name
from utils.data_generator import gaussian_circle
from config import SAVE_DIR

CLASS_ONE_COLOR = '#FFFFFF'
CLASS_TWO_COLOR = '#0097E8'
EDGE_COLOR = '#1A1A1A'


def plot_bar_graph(gpu_results, cpu_results, labels, save=False, show=False, save_folder=''):
    n_groups = len(labels)  # number of experiments run

    means_cpu = cpu_results
    means_gpu = gpu_results

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    plt.bar(index, means_cpu, bar_width, alpha=opacity, color='b', error_kw=error_config, label='cpu')
    plt.bar(index + bar_width, means_gpu, bar_width, alpha=opacity, color='r', error_kw=error_config, label='gpu')

    plt.xlabel('Experiment values')
    plt.ylabel('Time spent')
    plt.xticks(index + bar_width, labels)
    plt.legend()
    plt.tight_layout()
    if save:
        save_plot(fig, save_folder)
    if show:
        plt.show()


def plot_ant_scores(ant_scores, save=False, show=False, save_folder='', file_name=None):
    fig = plt.figure()
    fig.suptitle("ant scores")
    ax = fig.add_subplot(111)
    x = range(len(ant_scores))
    y = ant_scores
    ax.plot(x, y, 'k-')
    ax.axis([0, len(ant_scores), min(ant_scores), max(ant_scores)])
    if save:
        save_plot(fig, save_folder, file_name=file_name)
    if show:
        plt.show()


def plot_path_with_data(path, data, matrix, save=False, show=False, save_folder='', color='k-', file_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("off")
    plot_matrix(matrix, ax, with_vertices=False)
    plot_path(path, ax, color)
    plot_data(data, ax)
    if save:
        save_plot(fig, save_folder, eps=False, file_name=file_name)
    if show:
        plt.show()


def plot_multi(best_path, rest_path, data, matrix, save=False, show=False, save_folder=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis("off")
    plot_matrix(matrix, ax, with_vertices=False)
    plot_data(data, ax)
    plot_path(best_path, ax)
    plot_path(rest_path, ax, color='r-')
    if save:
        save_plot(fig, save_folder)
    if show:
        plt.show()


def plot_data(data, subplot=None, show=False):
    ax = subplot if subplot is not None else plt
    if data.shape[0] > 2:
        temp = data.T
        red = temp[temp[:, 2] == 0][:, :2].T
        blue = temp[temp[:, 2] == 1][:, :2].T
        ax.scatter(red[0], red[1], color=CLASS_ONE_COLOR, s=80, edgecolor=EDGE_COLOR, lw=1.0)
        ax.scatter(blue[0], blue[1], color=CLASS_TWO_COLOR, s=80, edgecolor=EDGE_COLOR, lw=1.0)
    else:
        ax.plot(data[0], data[1], 'o')
    ax.axis([np.amin(data[0]) - .2,
             np.amax(data[0]) + .2,
             np.amin(data[1]) - .2,
             np.amax(data[1]) + .2])

    if show:
        plt.show()


def plot_matrix(matrix, subplot=None, show=False, with_vertices=True, save=False):
    ax = subplot if subplot is not None else plt
    for edge in matrix.edges:
        ax.plot([edge.a.x, edge.b.x], [edge.a.y, edge.b.y], '--', color='#CFCFCF')
    if with_vertices:
        for i, v in enumerate(matrix.vertices):
            ax.plot(v.x, v.y, 'o', color='w')

    margin_x = (matrix.x_min_max[1] - matrix.x_min_max[0]) / 2
    margin_y = (matrix.y_min_max[1] - matrix.y_min_max[0]) / 2
    ax.axis([matrix.x_min_max[0] - margin_x, matrix.x_min_max[1] + margin_x,
             matrix.y_min_max[0] - margin_y, matrix.y_min_max[1] + margin_y])
    if show:
        plt.show()
    if save:
        save_plot(ax)


def plot_path(path, subplot, color='k-'):
    for edge in path:
        subplot.plot([edge.a.x, edge.b.x], [edge.a.y, edge.b.y], color, linewidth=3)


def plot_smooth_curves(curves, labels, y_axis_label='Score', show=False, loc='upper left'):
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.xlabel("Ants")
    plt.ylabel(y_axis_label)
    for i, c in enumerate(curves):
        window_size = c.size // 4
        if window_size % 2 == 0:
            window_size += 1
        y = savgol_filter(c, window_size, 2)
        ax.plot(range(y.shape[0]), y, label=labels[i])
    ax.legend(loc=loc)
    if show:
        plt.show()
    return f


def plot_curves(curves, labels, y_axis_label='Score', show=False, loc='upper left'):
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.xlabel("Ants")
    plt.ylabel(y_axis_label)
    for i, c in enumerate(curves):
        ax.plot(range(c.shape[0]), c, label=labels[i])
    ax.legend(loc=loc)
    if show:
        plt.show()
    return f


def plot_pheromones(matrix, data, tau_min, tau_max, file_name=None, save=False, folder_name='', show=False):
    min_val = 0.5
    max_val = 15.0

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Pheromones")
    for edge in matrix.edges:
        line = ax.plot([edge.a.x, edge.b.x], [edge.a.y, edge.b.y], 'k-')
        lw = edge.pheromone_strength*((max_val - min_val) / (tau_max - tau_min))
        plt.setp(line, lw=lw)

    if data is not None:
        plot_data(data)

    margin_x = (matrix.x_min_max[1] - matrix.x_min_max[0]) / 10
    margin_y = (matrix.y_min_max[1] - matrix.y_min_max[0]) / 10
    ax.axis([matrix.x_min_max[0] - margin_x,
             matrix.x_min_max[1] + margin_x,
             matrix.y_min_max[0] - margin_y,
             matrix.y_min_max[1] + margin_y])
    if save:
        save_plot(fig, folder_name, file_name=file_name, eps=False)
    if show:
        plt.show()
    plt.close(fig)


def save_plot(fig=None, parent_folder='', file_name=None, eps=True):
    if parent_folder != '':
        directory = os.path.join(SAVE_DIR, parent_folder)
    else:
        directory = generate_folder_name()
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = datetime.utcnow().strftime('%Y-%m-%d %H_%M_%S_%f')[:-5] if file_name is None else file_name
    if fig is None:
        fig = plt
    if eps:
        fig.savefig(os.path.join(directory, file_name + '.eps'))
    fig.savefig(os.path.join(directory, file_name + '.png'), transparent=False)


def hide_top_and_right_axis(ax):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def main():
    # points = uniform_rectangle((2, 4), (2, 4), 500)
    points = gaussian_circle(10.0, 500, 0)
    points2 = gaussian_circle(5.0, 500, 1)
    points = np.concatenate((points, points2), axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hide_top_and_right_axis(ax)
    # plt.axis('off')
    plot_data(points, ax)
    save_plot(fig)
    # plt.show()

if __name__ == "__main__":
    main()
