from itertools import product
import numpy as np


class AcocMatrix:
    def __init__(self, data, tau_initial=1.0, granularity=10):
        self.x_min_max = np.amin(data[0]) - .1, np.amax(data[0]) + .1
        self.y_min_max = np.amin(data[1]) - .1, np.amax(data[1]) + .1
        self.tau_initial = tau_initial

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=int(granularity), endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=int(granularity), endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = init_vertices(coordinates)
        # The number of edges |E| depends on number of vertices with the following formula |E| = 2(y(x-1) + x(y-1))
        # Example: A 4x4 matrix will generate 2(4(4-1) + 4(4-1)) = 48
        self.edges = init_edges(self.vertices, self.tau_initial)
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
        return self.x, self.y

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
        if (vertex.x, vertex.y) == e.start.__repr__():
            connected_edges.append(e)
    vertex.connected_edges = connected_edges


def init_vertices(coordinates):
    vertices = []

    for x, y in coordinates:
        vertices.append(Vertex(x, y))
    return vertices


def init_edges(vertices, tau_initial):
    edges = []

    for v in vertices:
        vertices_in_row = [p for p in vertices if p.x > v.x and p.y == v.y]
        if len(vertices_in_row) > 0:
            next_east = min(vertices_in_row, key=lambda pos: pos.x)
            e1 = AcocEdge(v, next_east, tau_initial)
            e2 = AcocEdge(next_east, v, tau_initial, e1)
            e1.twin = e2
            edges.extend([e1, e2])
        vertices_in_column = [p for p in vertices if p.y > v.y and p.x == v.x]
        if len(vertices_in_column) > 0:
            next_north = min(vertices_in_column, key=lambda pos: pos.y)
            e1 = AcocEdge(v, next_north, tau_initial)
            e2 = AcocEdge(next_north, v, tau_initial, e1)
            e1.twin = e2
            edges.extend([e1, e2])
    return edges
