from itertools import product
import matplotlib.pyplot as plt


class AcocMatrix:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.vertices = self.init_vertices(x_size, y_size)
        self.edges = self.init_matrix_edges(x_size, y_size)

    def init_matrix_edges(self, x_size, y_size):
        edges = []

        for x, y in self.vertices:
            if not x == x_size:
                east_neighbor = (x + 1, y)
                edges.append(AcocEdge((x, y), east_neighbor))
            if not y == y_size:
                south_neighbor = (x, y + 1)
                edges.append(AcocEdge((x, y), south_neighbor))
        return edges

    def init_vertices(self, x_size, y_size):
        return list(product(range(1, x_size + 1), range(1, y_size + 1)))

    def get_connected_edges(self, vertex):
        edges = []
        for edge in self.edges:
            if edge.a_vertex == vertex or edge.b_vertex == vertex:
                edges.append(edge)
        return edges

    def show_plot(self):
        x_coord, y_coord = zip(*self.vertices)
        for edge in self.edges:
            plt.plot([edge.a_vertex[0], edge.b_vertex[0]], [edge.a_vertex[1], edge.b_vertex[1]], 'k-')
        plt.plot(x_coord, y_coord, 'o')
        plt.axis([0, self.x_size + 1, 0, self.y_size + 1])
        plt.show()


class AcocEdge:
    def __init__(self, a_vertex, b_vertex, pheromone_strength=0.1):
        self.a_vertex = a_vertex
        self.b_vertex = b_vertex
        self.pheromone_strength = pheromone_strength

    def __repr__(self):
        return str((self.a_vertex, self.b_vertex, self.pheromone_strength))

    def have_vertices(self, vertex_a, vertex_b):
        if (self.a_vertex == vertex_a or self.a_vertex == vertex_b) and \
                (self.b_vertex == vertex_a or self.b_vertex == vertex_b):
            return True
        return False


if __name__ == "__main__":
    matrix = AcocMatrix(10, 10)
    matrix.show_plot()
