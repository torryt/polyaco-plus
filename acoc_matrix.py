from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import math

import acoc_plotter


class AcocMatrix:
    def __init__(self, data, q_initial=0.1, granularity=1.0):
        self.x_min_max = int(np.amin(data[0]) - 1), int(np.amax(data[0]) + 2)
        self.y_min_max = int(np.amin(data[1]) - 1), int(np.amax(data[1]) + 2)
        self.q_initial = q_initial

        grid_width = math.ceil(self.x_min_max[1]) - int(self.x_min_max[0]) + 1
        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=grid_width*granularity, endpoint=True)
        grid_height = math.ceil(self.y_min_max[1]) - int(self.y_min_max[0]) + 1
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=grid_height*granularity, endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.edges = init_edges(coordinates, self.q_initial)
        self.vertices = init_vertices(coordinates, self.edges)

    def plot_matrix(self, show=True):
        x_coord, y_coord = zip(*[(v.x, v.y) for v in self.vertices])
        for edge in self.edges:
            plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], '--', color='#CFCFCF')

        # plt.plot(x_coord, y_coord, 'o', color='w')
        for i, v in enumerate(self.vertices):
            if i % 2 == 0:
                plt.plot(v.x, v.y, 'o', color='w')
            else:
                plt.plot(v.x, v.y, 'o', color='k')

        plt.axis([self.x_min_max[0] - 1, self.x_min_max[1], self.y_min_max[0] - 1, self.y_min_max[1]])
        if show:
            plt.show()

    def find_vertex(self, x_y):
        for v in self.vertices:
            if (v.x, v.y) == x_y:
                return v
        return None


class AcocEdge:
    def __init__(self, vertex_a, vertex_b, pheromone_strength=0.1):
        self.vertex_a = vertex_a
        self.vertex_b = vertex_b
        self.pheromone_strength = pheromone_strength

    def __repr__(self):
        return str((self.vertex_a, self.vertex_b, self.pheromone_strength))

    def has_vertex(self, vertex):
        if self.vertex_a == vertex or self.vertex_b == vertex:
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
        if ((vertex.x, vertex.y) == e.vertex_a.coordinates()) or ((vertex.x, vertex.y) == e.vertex_b.coordinates()):
            connected_edges.append(e)
    vertex.connected_edges = connected_edges


def init_vertices(coordinates, edges):
    vertices = [Vertex(p[0], p[1]) for p in coordinates]

    for v in vertices:
        connect_edges_to_vertex(v, edges)
    return vertices


def init_edges(coordinates, q_initial):
    edges = []

    for x, y in coordinates:
        points_in_row = [p for p in coordinates if p[0] > x]
        if len(points_in_row) > 0:
            next_east = min(points_in_row, key=lambda pos: pos[0])
            east_neighbor = Vertex(next_east[0], y)
            edges.append(AcocEdge(Vertex(x, y), east_neighbor, q_initial))

        points_in_column = [p for p in coordinates if p[1] > y]
        if len(points_in_column) > 0:
            next_north = min(points_in_column, key=lambda pos: pos[1])
            north_neighbor = Vertex(x, next_north[1])
            edges.append(AcocEdge(Vertex(x, y), north_neighbor, q_initial))

    return edges


def main():
    from data_generator import uniform_rectangle

    red = np.insert(uniform_rectangle((1, 3), (2, 4), 500), 2, 0, axis=0)
    blue = np.insert(uniform_rectangle((4, 6), (2, 4), 500), 2, 1, axis=0)
    data = np.concatenate((red, blue), axis=1)

    matrix = AcocMatrix(data, granularity=1.0)
    matrix.plot_matrix(show=False)
    acoc_plotter.plot_data(data, show=False)
    plt.axis([matrix.x_min_max[0] - 1, matrix.x_min_max[1] + 1, matrix.y_min_max[0] - 1, matrix.y_min_max[1] + 1])
    # plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
