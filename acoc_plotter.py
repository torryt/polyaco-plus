from matplotlib import pyplot as plt
import numpy as np
from data_generator import uniform_rectangle, uniform_circle
from scipy.interpolate import interp1d


class LivePheromonePlot:
    def __init__(self, matrix, data=None):
        plt.ion()
        self.plot_lines = []
        for edge in matrix.edges:
            line = plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
            plt.setp(line, linewidth=edge.pheromone_strength)
            self.plot_lines.append(line)

        if data is not None:
            plot_data(data)

        plt.axis([matrix.x_min_max[0] - 1, matrix.x_min_max[1], matrix.y_min_max[0] - 1, matrix.y_min_max[1]])
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
                    plt.setp(self.plot_lines[j], linewidth=5.0, color='b')
                elif edge == current_edge:
                    plt.setp(self.plot_lines[j], linewidth=5.0, color='r')
                else:
                    plt.setp(self.plot_lines[j], linewidth=edge.pheromone_strength, color='k')

        else:
            for j, edge in enumerate(new_edges):
                plt.setp(self.plot_lines[j], linewidth=edge.pheromone_strength)
        plt.draw()
        plt.pause(0.01)


def plot_ant_scores(ant_scores):
    plt.figure(2)
    x = range(len(ant_scores))
    y = ant_scores
    plt.plot(x, y, 'k-')
    plt.axis([0, len(ant_scores), min(ant_scores), max(ant_scores)])
    plt.title("Ant Scores")
    plt.show()


def plot_path(path, matrix):
    for edge in path:
        plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
    plt.axis([-1, matrix.x_size, -1, matrix.y_size])


def plot_path_with_data(path, data):
    plt.close()
    for edge in path:
        plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
    plot_data(data)
    plt.show()


def plot_pheromone_values(matrix, show=False):
    for edge in matrix.edges:
        line = plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
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
    plt.plot(x_coord2, y_coord2, 'r')
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


def plot_data(data, show=True):
    if data.shape[0] > 2:
        temp = data.T
        red = temp[temp[:, 2] == 0][:, :2].T
        blue = temp[temp[:, 2] == 1][:, :2].T
        plt.scatter(red[0], red[1], color='r', s=100, edgecolor='k')
        plt.scatter(blue[0], blue[1], color='b', s=100, edgecolor='k')
    else:
        plt.plot(data[0], data[1], 'o')
    plt.axis([np.amin(data[0]) - 1, np.amax(data[0]) + 1, np.amin(data[1]) - 1, np.amax(data[1]) + 1])

    if show:
        plt.show()


if __name__ == "__main__":
    def main():
        # points = uniform_rectangle((2, 4), (2, 4), 500)
        points = uniform_circle(10.0, 500, 0)
        points2 = uniform_circle(5.0, 500, 1)
        points = np.concatenate((points, points2), axis=1)
        plot_data(points)

    main()
