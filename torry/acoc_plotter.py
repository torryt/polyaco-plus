from matplotlib import pyplot as plt
import numpy as np


class LivePheromonePlot:
    def __init__(self, matrix, start_coordinates=None, target_coordinates=None):
        plt.ion()
        self.plot_lines = []
        for edge in matrix.edges:
            line = plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
            plt.setp(line, linewidth=edge.pheromone_strength)
            self.plot_lines.append(line)

        if start_coordinates and target_coordinates:
            plt.plot([start_coordinates[0], target_coordinates[0]], [start_coordinates[1], target_coordinates[1]], 'o')

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


def plot_path_lengths(path_lengths):
    x_coord = range(len(path_lengths))
    y_coord = path_lengths
    plt.plot(x_coord, y_coord, 'k-')
    plt.axis([0, len(path_lengths), 0, max(path_lengths)])


def plot_path(path, matrix):
    for edge in path:
        plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
    plt.axis([-1, matrix.x_size, -1, matrix.y_size])


def plot_path_with_data(path, data):
    plt.close()
    for edge in path:
        plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
    plt.plot(data[0], data[1], 'o')
    plt.axis([np.amin(data[0]) - 1, np.amax(data[0]) + 1, np.amin(data[1]) - 1, np.amax(data[1]) + 1])


def plot_pheromone_values(matrix, last_edge=None):
    for edge in matrix.edges:
        line = plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
        plt.setp(line, linewidth=edge.pheromone_strength)
    plt.axis([-1, matrix.x_size, -1, matrix.y_size])


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
        plt.figure(2)
        plt.title("ACO Path Lengths")
        plot_path_lengths(ant_path_lengths)
    plt.show()


def plot_aco_and_random(aco_path_lengths, random_path_lengths):
    plt.figure(2)
    plot_two_path_lengths(aco_path_lengths, random_path_lengths)
    plt.show()


def plot_data(data):
    plt.plot(data[0], data[1], 'o')
    plt.axis([np.amin(data[0]) - 1, np.amax(data[0]) + 1, np.amin(data[1]) - 1, np.amax(data[1]) + 1])
    plt.show()