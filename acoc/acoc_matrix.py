from itertools import product

import numpy as np
import math

from acoc.edge import Edge
from acoc.vertex import Vertex


class AcocMatrix:
    def __init__(self, data, tau_initial=1.0, granularity=10, old_matrix=None):
        self.x_min_max = np.amin(data[0]) - .1, np.amax(data[0]) + .1
        self.y_min_max = np.amin(data[1]) - .1, np.amax(data[1]) + .1
        self.tau_initial = tau_initial
        self.level = 1
        self.granularity = granularity

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=int(self.granularity), endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=int(self.granularity), endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = [Vertex(x, y) for x, y in coordinates]

        # The number of edges |E| depends on number of vertices with the following formula |E| = y(x-1) + x(y-1)
        # Example: A 4x4 matrix will generate 4(4-1) + 4(4-1) = 24
        self.edges = create_edges(self.vertices, self.tau_initial)
        [connect_edges_to_vertex(v, self.edges) for v in self.vertices]

    def add_to_plot(self, ax):
        for edge in self.edges:
            ax.plot([edge.a.x, edge.b.x], [edge.a.y, edge.b.y], '--', color='#CFCFCF')
        for i, v in enumerate(self.vertices):
            if i % 2 == 0:
                ax.plot(v.x, v.y, 'o', color='w')
            else:
                ax.plot(v.x, v.y, 'o', color='k')
        ax.axis([self.x_min_max[0] - 1, self.x_min_max[1] + 1, self.y_min_max[0] - 1, self.y_min_max[1] + 1])

    def level_up(self):
        new_edges = []
        y_step = (self.y_min_max[1] - self.y_min_max[0]) / ((self.granularity - 1) * 2)
        x_step = (self.x_min_max[1] - self.x_min_max[0]) / ((self.granularity - 1) * 2)
        remove_edges = []
        for i, e in enumerate(self.edges):
            v = Vertex(((e.a.x + e.b.x)/2), ((e.a.y + e.b.y)/2))
            self.vertices.append(v)
            if math.isclose(v.y, e.a.y) and v.y < self.y_min_max[1]:
                self.vertices.append(Vertex(v.x, v.y + y_step))

            e1 = Edge(e.a, v, e.pheromone_strength)
            e2 = Edge(v, e.b, e.pheromone_strength)
            remove_edges.append(e)
            new_edges.extend([e1, e2])
        self.edges.extend(new_edges)
        [self.edges.remove(e) for e in remove_edges]

        [connect_edges_to_vertex(v, self.edges) for v in self.vertices]

        unconnected_vertices = list(filter(lambda v: len(v.connected_edges) == 0, mtrx.vertices))
        for v in unconnected_vertices:
            self.edges.append(Edge(v, find_vertex(v.x, v.y + y_step, self.vertices)))   # UP
            self.edges.append(Edge(find_vertex(v.x, v.y - y_step, self.vertices), v))   # DOWN
            self.edges.append(Edge(find_vertex(v.x - x_step, v.y, self.vertices), v))   # LEFT
            self.edges.append(Edge(v, find_vertex(v.x + x_step, v.y, self.vertices)))   # RIGHT
        [connect_edges_to_vertex(v, self.edges) for v in self.vertices]


def find_vertex(x, y, vertices):
    try:
        return [v for v in vertices if v == Vertex(x, y)][0]
    except IndexError:
        pass


def connect_edges_to_vertex(vertex, edges):
    vertex.connected_edges = [e for e in edges if vertex == e.a or vertex == e.b]


def create_edges(vertices, tau_initial):
    edges = []

    for vertex in vertices:
        vertices_in_row = [p for p in vertices if p.x > vertex.x and p.y == vertex.y]
        if len(vertices_in_row) > 0:
            next_east = min(vertices_in_row, key=lambda pos: pos.x)
            edge = Edge(vertex, next_east, tau_initial)
            edges.append(edge)
        vertices_in_column = [p for p in vertices if p.y > vertex.y and p.x == vertex.x]
        if len(vertices_in_column) > 0:
            next_north = min(vertices_in_column, key=lambda pos: pos.y)
            edge = Edge(vertex, next_north, tau_initial)
            edges.append(edge)
    return edges


if __name__ == "__main__":
    from acoc.acoc_plotter import plot_matrix
    mtrx = AcocMatrix([[[0, 0]], [[1, 1]]], granularity=2)
    plot_matrix(mtrx, show=True)
    mtrx.level_up()
    plot_matrix(mtrx, show=True)
    mtrx.level_up()
    print("Edges: {}, vertices: {}".format(len(mtrx.edges), len(mtrx.vertices)))
    plot_matrix(mtrx, show=True)
