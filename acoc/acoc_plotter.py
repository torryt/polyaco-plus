from __future__ import division
import os
import uuid
from time import strftime
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from utils.data_generator import uniform_circle

from config import SAVE_DIR
BLUE_COLOR = '#0097E8'
RED_COLOR = '#F03A3A'
EDGE_COLOR = '#1A1A1A'



class LivePheromonePlot:
    def __init__(self, matrix, data=None):
        plt.ion()
        self.plot_lines = []
        for edge in matrix.edges:
            line = plt.plot([edge.start.x, edge.target.x], [edge.start.y, edge.target.y], 'k-')
            plt.setp(line, linewidth=edge.pheromone_strength)
            self.plot_lines.append(line)

        if data is not None:
            plot_data(data)

        plt.axis([matrix.x_min_max[0] - .1,
                  matrix.x_min_max[1] + .1,
                  matrix.y_min_max[0] - .1,
                  matrix.y_min_max[1] + .1])
        plt.draw()
        plt.pause(0.01)

    @staticmethod
    def close():
        plt.clf()
        plt.ioff()

    def update(self, new_edges, current_edge=None, connected_edges=None):
        if current_edge:
            for j, edge in enumerate(new_edges):
                if edge in connected_edges:
                    plt.setp(self.plot_lines[j], linewidth=5.0, color=BLUE_COLOR)
                elif edge == current_edge:
                    plt.setp(self.plot_lines[j], linewidth=5.0, color=RED_COLOR)
                else:
                    plt.setp(self.plot_lines[j], linewidth=edge.pheromone_strength, color='k')

        else:
            for j, edge in enumerate(new_edges):
                plt.setp(self.plot_lines[j], linewidth=edge.pheromone_strength)
        plt.draw()
        plt.pause(0.01)


def plot_ant_scores(ant_scores, save=False, show=False):
    plt.close()
    x = range(len(ant_scores))
    y = ant_scores
    plt.plot(x, y, 'k-')
    plt.axis([0, len(ant_scores), min(ant_scores), max(ant_scores)])
    plt.title("Ant Scores")
    if save:
        save_plot()
    if show:
        plt.show()


def plot_path_with_data(path, data, save=False, show=False):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    hide_top_and_right_axis(ax)
    plot_data(data, ax)
    plot_path(path, ax)
    if save:
        save_plot(fig)
    if show:
        plt.show()


def plot_pheromone_values(matrix, show=False):
    for edge in matrix.edges:
        line = plt.plot([edge.start.x, edge.target.x], [edge.start.y, edge.target.y], 'k-')
        plt.setp(line, linewidth=edge.pheromone_strength)
    plt.axis([matrix.x_min_max[0], matrix.x_min_max[1], matrix.y_min_max[0], matrix.y_min_max[1]])
    if show:
        plt.show()


def plot_two_path_lengths(path_length1, path_length2):
    x_coord = range(len(path_length1))
    y_coord = path_length1

    x_coord2 = range(len(path_length2))
    y_coord2 = path_length2

    plt.plot(x_coord, y_coord, 'g')
    plt.plot(x_coord2, y_coord2, RED_COLOR)
    plt.axis([0, len(path_length1), 0, max(path_length1)])


def draw_all(ant_path_lengths, shortest_path, data, ran_path_lengths=None):
    plt.figure(1)
    plt.subplot(211)
    plt.title("Path")
    plot_path_with_data(shortest_path, data)
    plt.subplot(212)
    plt.title("Pheromone Values")

    if ran_path_lengths:
        plot_aco_and_random(ant_path_lengths, ran_path_lengths)
    else:
        plt.figure(1)
        plt.title("ACO Path Lengths")
        plot_ant_scores(ant_path_lengths)
    plt.show()


def plot_aco_and_random(aco_path_lengths, random_path_lengths):
    plt.figure(1)
    plot_two_path_lengths(aco_path_lengths, random_path_lengths)
    plt.show()


def plot_data(data, subplot=None, show=False):
    if subplot is not None:
        ax = subplot
    else:
        ax = plt
    if data.shape[0] > 2:
        temp = data.T
        red = temp[temp[:, 2] == 0][:, :2].T
        blue = temp[temp[:, 2] == 1][:, :2].T
        ax.scatter(red[0], red[1], color=RED_COLOR, s=80, edgecolor=EDGE_COLOR)
        ax.scatter(blue[0], blue[1], color=BLUE_COLOR, s=80, edgecolor=EDGE_COLOR)
    else:
        ax.plot(data[0], data[1], 'o')
    ax.axis([np.amin(data[0]) - 1, np.amax(data[0]) + 1, np.amin(data[1]) - 1, np.amax(data[1]) + 1])

    if show:
        plt.show()


def plot_path(path, subplot):
    for edge in path:
        subplot.plot([edge.start.x, edge.target.x], [edge.start.y, edge.target.y], 'k-')


def plot_smooth_curves(curves, labels, show=False):
    f = plt.figure()
    ax = f.add_subplot(111)
    for i, c in enumerate(curves):
        window_size = c.size // 4
        if window_size % 2 == 0:
            window_size += 1
        y = savgol_filter(c, window_size, 2)
        ax.plot(range(y.shape[0]), y, label=labels[i])
    ax.legend()
    if show:
        plt.show()
    return f


def save_plot(fig=None, parent_folder=''):
    directory = SAVE_DIR + strftime("%Y-%m-%d_%H%M") + parent_folder + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = str(uuid.uuid4())
    if fig is None:
        plt.savefig(os.path.join(directory, file_name + '.eps'))
        plt.savefig(os.path.join(directory, file_name + '.png'))
    else:
        fig.savefig(os.path.join(directory, file_name + '.eps'))
        fig.savefig(os.path.join(directory, file_name + '.png'))


def hide_top_and_right_axis(ax):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


if __name__ == "__main__":
    def main():
        # points = uniform_rectangle((2, 4), (2, 4), 500)
        points = uniform_circle(10.0, 500, 0)
        points2 = uniform_circle(5.0, 500, 1)
        points = np.concatenate((points, points2), axis=1)
        plot_data(points)

    main()
