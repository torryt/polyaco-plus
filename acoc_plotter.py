from matplotlib import pyplot as plt
import numpy as np


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
    x_coord = range(len(ant_scores))
    y_coord = ant_scores
    plt.plot(x_coord, y_coord, 'k-')
    plt.axis([0, len(ant_scores), 0, max(ant_scores)])
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

# Used for displaying shortest path + random


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
        plt.plot(red[0], red[1], 'o', color='r')
        plt.plot(blue[0], blue[1], 'o', color='b')
    else:
        plt.plot(data[0], data[1], 'o')
    plt.axis([np.amin(data[0]) - 1, np.amax(data[0]) + 1, np.amin(data[1]) - 1, np.amax(data[1]) + 1])

    if show:
        plt.show()

if __name__ == "__main__":
    def main():
        from data_generator import uniform_rectangle
        points = uniform_rectangle((2, 4), (2, 4), 500)
        plot_data(points)


    main()
