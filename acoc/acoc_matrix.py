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

    def level_up(self, polygon):
        self.level += 1
        new_edges = []
        self.granularity = self.granularity * 2 - 1
        edge_y_length = (self.y_min_max[1] - self.y_min_max[0]) / (self.granularity - 1)
        edge_x_length = (self.x_min_max[1] - self.x_min_max[0]) / (self.granularity - 1)
        remove_edges = []
        vertices_without_connecting_edges = []
        for e in self.edges:
            mid_v = Vertex(((e.a.x + e.b.x) / 2), ((e.a.y + e.b.y) / 2))
            self.vertices.append(mid_v)

            if math.isclose(mid_v.y, e.a.y) and mid_v.y < self.y_min_max[1]:
                v = Vertex(mid_v.x, mid_v.y + edge_y_length)
                self.vertices.append(v)
                vertices_without_connecting_edges.append(v)

            new_e1 = Edge(e.a, mid_v, e.pheromone_strength)
            new_e2 = Edge(mid_v, e.b, e.pheromone_strength)
            remove_edges.append(e)
            new_edges.extend([new_e1, new_e2])
            if e in polygon:
                polygon.remove(e)
                polygon.extend([new_e1, new_e2])
        self.edges.extend(new_edges)
        [self.edges.remove(e) for e in remove_edges]

        for v in vertices_without_connecting_edges:
            directions = [find_vertex(v.x, v.y + edge_y_length, self.vertices),
                          find_vertex(v.x, v.y - edge_y_length, self.vertices),
                          find_vertex(v.x - edge_x_length, v.y, self.vertices),
                          find_vertex(v.x + edge_x_length, v.y, self.vertices)]
            for d in directions:
                if d is not None:
                    self.edges.append(Edge(v, d, pheromone_strength=self.tau_initial))
                else:
                    pass
        [connect_edges_to_vertex(v, self.edges) for v in self.vertices]
        return polygon


def find_vertex(x, y, vertices):
    try:
        return filter(lambda v: v == Vertex(x, y), vertices).__next__()
    except StopIteration:
        return None


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
    from acoc.acoc_plotter import plot_pheromones
    from utils import generate_folder_name
    data = np.array([[[0, 0]], [[1, 1]]])
    mtrx = AcocMatrix(data, granularity=3)
    pol = mtrx.edges[:4]
    mtrx.edges[1].pheromone_strength = 5
    mtrx.edges[0].pheromone_strength = 3
    save_folder = generate_folder_name()

    plot_pheromones(mtrx, data, tau_min=1, tau_max=10, folder_name=save_folder, save=True)
    for i in range(4):
        print("Polygon length before: {}".format(len(pol)))
        mtrx.level_up(pol)
        print("Polygon length after: {}".format(len(pol)))
        print("Edges: {}, vertices: {}".format(len(mtrx.edges), len(mtrx.vertices)))
        if i > 3:
            plot_pheromones(mtrx, data, tau_min=1, tau_max=10, show=True)
        else:
            plot_pheromones(mtrx, data, tau_min=1, tau_max=10, folder_name=save_folder, save=True)
