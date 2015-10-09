from matplotlib import pyplot as plt

__author__ = 'torrytufteland'


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