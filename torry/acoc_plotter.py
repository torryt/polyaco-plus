from matplotlib import pyplot as plt


class LivePheromonePlot:
    def __init__(self, matrix, start_coordinates, target_coordinates):
        plt.ion()
        self.plot_lines = []
        for edge in matrix.edges:
            line = plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
            plt.setp(line, linewidth=edge.pheromone_strength)
            self.plot_lines.append(line)
        plt.plot([start_coordinates[0], target_coordinates[0]], [start_coordinates[1], target_coordinates[1]], 'o')

        plt.axis([-1, matrix.x_size, -1, matrix.y_size])
        plt.draw()
        plt.pause(0.01)

    @staticmethod
    def close():
        plt.clf()
        plt.ioff()

    def update(self, new_edges):
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


def plot_pheromone_values(matrix):
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


def draw_all(all_paths, shortest_path, matrix, random, ran_path_lengths=None):
    plt.figure(1)
    plt.subplot(211)
    plt.title("Path")
    plot_path(shortest_path, matrix)
    plt.subplot(212)
    plt.title("Pheromone Values")
    plot_pheromone_values(matrix)

    if not random:
        plt.figure(2)
        plt.title("ACO Path Lengths")
        plot_path_lengths(all_paths)
    else:
        plot_aco_and_random(all_paths, ran_path_lengths)
    plt.show()


def plot_aco_and_random(aco_path_lengths, random_path_lengths):
    plt.figure(2)
    plot_two_path_lengths(aco_path_lengths, random_path_lengths)
    plt.show()
