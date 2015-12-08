from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from acoc import acoc_plotter
from utils import data_generator as dg


class AcocMatrix:
    def __init__(self, data, q_initial=1.0, granularity=10):
        self.x_min_max = np.amin(data[0]) - .1, np.amax(data[0]) + .1
        self.y_min_max = np.amin(data[1]) - .1, np.amax(data[1]) + .1
        self.q_initial = q_initial

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=int(granularity), endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=int(granularity), endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = init_vertices(coordinates)
        self.edges = init_edges(self.vertices, self.q_initial)
        [connect_edges_to_vertex(v, self.edges) for v in self.vertices]

    def add_to_plot(self, ax):
        for edge in self.edges:
            ax.plot([edge.start.x, edge.target.x], [edge.start.y, edge.target.y], '--', color='#CFCFCF')
        for i, v in enumerate(self.vertices):
            if i % 2 == 0:
                ax.plot(v.x, v.y, 'o', color='w')
            else:
                ax.plot(v.x, v.y, 'o', color='k')
        ax.axis([self.x_min_max[0] - 1, self.x_min_max[1] + 1, self.y_min_max[0] - 1, self.y_min_max[1] + 1])

    def find_vertex(self, x_y):
        for v in self.vertices:
            if (v.x, v.y) == x_y:
                return v
        return None


class AcocEdge:
    def __init__(self, start, target, pheromone_strength=0.1, twin=None):
        self.start = start
        self.target = target
        self.twin = twin
        self.pheromone_strength = pheromone_strength

    def __repr__(self):
        return str((self.start, self.target, self.pheromone_strength))

    def has_vertex(self, vertex):
        if self.start == vertex or self.target == vertex:
            return True
        return False


class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connected_edges = []

    def __repr__(self):
        return str((self.x, self.y))

    def coordinates(self):
        return self.x, self.y


def edge_is_in_list(edge, ignore_list):
    for e in ignore_list:
        if e == edge:
            return True
    return False


def connect_edges_to_vertex(vertex, edges):
    connected_edges = []
    for e in edges:
        if (vertex.x, vertex.y) == e.start.coordinates():
            connected_edges.append(e)
    vertex.connected_edges = connected_edges


def init_vertices(coordinates):
    vertices = []

    for x, y in coordinates:
        vertices.append(Vertex(x, y))
    return vertices
    # for e in edges:
    #     if e.start not in vertices:
    #         vertices.append(e.start)
    #     if e.target not in vertices:
    #         vertices.append(e.target)
    # # vertices = [Vertex(p[0], p[1]) for p in coordinates]
    #
    # for v in vertices:
    #     connect_edges_to_vertex(v, edges)
    # return vertices


def init_edges(vertices, q_initial):
    edges = []

    for v in vertices:
        vertices_in_row = [p for p in vertices if p.x > v.x and p.y == v.y]
        if len(vertices_in_row) > 0:
            next_east = min(vertices_in_row, key=lambda pos: pos.x)
            e1 = AcocEdge(v, next_east, q_initial)
            e2 = AcocEdge(next_east, v, q_initial, e1)
            e1.twin = e2
            edges.extend([e1, e2])
        vertices_in_column = [p for p in vertices if p.y > v.y and p.x == v.x]
        if len(vertices_in_column) > 0:
            next_north = min(vertices_in_column, key=lambda pos: pos.y)
            e1 = AcocEdge(v, next_north, q_initial)
            e2 = AcocEdge(next_north, v, q_initial, e1)
            e1.twin = e2
            edges.extend([e1, e2])
    return edges


def main():
    red = dg.gaussian_circle(1.0, 500, 1, .5)
    blue = dg.gaussian_circle(2.0, 500, 0, .5)
    data = np.concatenate((red, blue), axis=1)
    matrix = AcocMatrix(data, granularity=5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # acoc_plotter.plot_data(data, ax)
    acoc_plotter.plot_matrix(matrix, ax)
    ax.axis([matrix.x_min_max[0] - .1, matrix.x_min_max[1] + .1, matrix.y_min_max[0] - .1, matrix.y_min_max[1] + .1])
    # acoc_plotter.hide_top_and_right_axis(ax)
    plt.axis("off")
    acoc_plotter.save_plot(fig)
    # plt.show()

if __name__ == "__main__":
    main()
