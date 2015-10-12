from matplotlib import pyplot as plt


class LivePheromonePlot:
    def __init__(self, matrix, start_vertex, target_vertex):
        plt.ion()
        self.plot_lines = []
        for edge in matrix.edges:
            line = plt.plot([edge.a_vertex[0], edge.b_vertex[0]], [edge.a_vertex[1], edge.b_vertex[1]], 'k-')
            plt.setp(line, linewidth=edge.pheromone_strength)
            self.plot_lines.append(line)
        plt.plot([start_vertex[0], target_vertex[0]], [start_vertex[1], target_vertex[1]], 'o')

        plt.axis([-1, matrix.x_size, -1, matrix.y_size])
        plt.draw()
        plt.pause(0.01)

    def close(self):
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
    plt.show()


def plot_path(path, matrix):
    for edge in path:
        plt.plot([edge.a_vertex[0], edge.b_vertex[0]], [edge.a_vertex[1], edge.b_vertex[1]], 'k-')
    plt.axis([-1, matrix.x_size, -1, matrix.y_size])


def plot_pheromone_values(matrix):
    for edge in matrix.edges:
        line = plt.plot([edge.a_vertex[0], edge.b_vertex[0]], [edge.a_vertex[1], edge.b_vertex[1]], 'k-')
        plt.setp(line, linewidth=edge.pheromone_strength)
    plt.axis([-1, matrix.x_size, -1, matrix.y_size])

def draw_all(all_paths, shortest_path, matrix):
    plt.figure(1)
    plt.subplot(211)
    plt.title("Path")
    plot_path(shortest_path, matrix)
    plt.subplot(212)
    plt.title("Pheromone Values")
    plot_pheromone_values(matrix)

    plt.figure(2)
    plt.title("Path Lengths")
    plot_path_lengths(all_paths)
    plt.show()
