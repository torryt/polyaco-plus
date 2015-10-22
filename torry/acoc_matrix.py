from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import torry.acoc_plotter as acoc_plotter


class AcocMatrix:
    def __init__(self, data, blocked_edge_indexes=None):
        self.x_min_max = int(np.amin(data[0]) - 1), int(np.amax(data[0]) + 3)
        self.y_min_max = int(np.amin(data[1]) - 1), int(np.amax(data[1]) + 3)

        coordinates = list(product(range(self.x_min_max[0], self.x_min_max[1]),
                                   range(self.y_min_max[0], self.y_min_max[1])))
        self.edges = init_edges(self.x_min_max[1], self.y_min_max[1], coordinates, blocked_edge_indexes)
        self.vertices = init_vertices(coordinates, self.edges)

    def show_plot(self):
        x_coord, y_coord = zip(*[(v.x, v.y) for v in self.vertices])
        for edge in self.edges:
            plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
        plt.plot(x_coord, y_coord, 'o')
        plt.axis([self.x_min_max[0] - 1, self.x_min_max[1], self.y_min_max[0] - 1, self.y_min_max[1]])
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


def init_edges(x_size, y_size, coordinates, blocked_edge_indexes=None):
    edges = []
    for x, y in coordinates:
        if x != x_size - 1:
            east_neighbor = Vertex(x + 1, y)
            edges.append(AcocEdge(Vertex(x, y), east_neighbor))
        if y != y_size - 1:
            north_neighbor = Vertex(x, y + 1)
            edges.append(AcocEdge(Vertex(x, y), north_neighbor))
    if blocked_edge_indexes:
        blocked_edge_indexes.sort(reverse=True)
        for i in blocked_edge_indexes:
            edges.pop(i)
    return edges


def main():
    from torry.data_generator import uniform_rectangle
    red = np.insert(uniform_rectangle((2, 4), (2, 4), 10), 2, 0, axis=0)
    blue = np.insert(uniform_rectangle((6, 8), (2, 4), 10), 2, 1, axis=0)
    data = np.concatenate((red, blue), axis=1)

    matrix = AcocMatrix(data)
    acoc_plotter.plot_data(data)

    matrix.show_plot()

if __name__ == "__main__":
    main()
