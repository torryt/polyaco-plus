from itertools import product
import matplotlib.pyplot as plt


def init_vertices(x_size, y_size):
    return list(product(range(0, x_size), range(0, y_size)))


def edge_is_in_list(edge, ignore_list):
    for e in ignore_list:
        if e == edge:
            return True
    return False


class AcocMatrix:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.vertices = init_vertices(x_size, y_size)
        self.edges = self.init_matrix_edges(x_size, y_size)

    def init_matrix_edges(self, x_size, y_size):
        edges = []

        for x, y in self.vertices:
            if x != x_size - 1:
                east_neighbor = (x + 1, y)
                edges.append(AcocEdge((x, y), east_neighbor))
            if y != y_size - 1:
                north_neighbor = (x, y + 1)
                edges.append(AcocEdge((x, y), north_neighbor))
        return edges

    def get_connected_edges(self, vertex, ignore_list):
        edges = []
        for edge in self.edges:
            if (edge.a_vertex == vertex or edge.b_vertex == vertex) and not edge_is_in_list(edge, ignore_list):
                edges.append(edge)
        return edges

    def show_plot(self, edges):
        x_coord, y_coord = zip(*self.vertices)
        for edge in edges:
            plt.plot([edge.a_vertex[0], edge.b_vertex[0]], [edge.a_vertex[1], edge.b_vertex[1]], 'k-')
        plt.plot(x_coord, y_coord, 'o')
        plt.axis([min(x_coord) - 1, self.x_size, min(y_coord) - 1, self.y_size])
        plt.show()


class AcocEdge:
    def __init__(self, a_vertex, b_vertex, pheromone_strength=0.1):
        self.a_vertex = a_vertex
        self.b_vertex = b_vertex
        self.pheromone_strength = pheromone_strength

    def __repr__(self):
        return str((self.a_vertex, self.b_vertex, self.pheromone_strength))

    def has_both_vertices(self, vertex_a, vertex_b):
        if (self.a_vertex == vertex_a or self.a_vertex == vertex_b) and \
                (self.b_vertex == vertex_a or self.b_vertex == vertex_b):
            return True
        return False

    def has_vertex(self, vertex):
        if self.a_vertex == vertex or self.b_vertex == vertex:
            return True
        return False


if __name__ == "__main__":
    matrix = AcocMatrix(10, 10)
    matrix.show_plot(matrix.edges)
