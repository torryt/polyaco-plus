from itertools import product

import numpy as np

from acoc.edge import Edge
from acoc.vertex import Vertex


class AcocMatrix:
    def __init__(self, data, tau_initial=1.0, granularity=10):
        self.x_min_max = np.amin(data[0]) - .1, np.amax(data[0]) + .1
        self.y_min_max = np.amin(data[1]) - .1, np.amax(data[1]) + .1
        self.tau_initial = tau_initial

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=int(granularity), endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=int(granularity), endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = [Vertex(x, y) for x, y in coordinates]

        # The number of edges |E| depends on number of vertices with the following formula |E| = y(x-1) + x(y-1)
        # Example: A 4x4 matrix will generate 4(4-1) + 4(4-1) = 24
        self.edges = init_edges(self.vertices, self.tau_initial)

        def connect_edges_to_vertex(vertex, edges):
            vertex.connected_edges = [e for e in edges if vertex == e.a or vertex == e.b]
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


def init_edges(vertices, tau_initial):
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
    plot_matrix(AcocMatrix([[[0, 0]], [[1, 1]]]), show=True)
