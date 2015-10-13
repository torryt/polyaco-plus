from itertools import product
import matplotlib.pyplot as plt


def edge_is_in_list(edge, ignore_list):
    for e in ignore_list:
        if e == edge:
            return True
    return False


def init_edges(x_size, y_size, coordinates):
    edges = []
    for x, y in coordinates:
        if x != x_size - 1:
            east_neighbor = Vertex(x + 1, y)
            edges.append(AcocEdge(Vertex(x, y), east_neighbor))
        if y != y_size - 1:
            north_neighbor = Vertex(x, y + 1)
            edges.append(AcocEdge(Vertex(x, y), north_neighbor))
    return edges


def connect_edges_to_vertex(vertex, edges):
    connected_edges = []
    for e in edges:
        if ((vertex.x, vertex.y) == e.vertex_a.coordinates()) or ((vertex.x, vertex.y) == e.vertex_b.coordinates()):
            connected_edges.append(e)
    vertex.connected_edges = connected_edges


def init_matrix(x_size, y_size):
    points = list(product(range(0, x_size), range(0, y_size)))
    vertices = [Vertex(p[0], p[1]) for p in points]
    edges = init_edges(x_size, y_size, points)

    for v in vertices:
        connect_edges_to_vertex(v, edges)
    return vertices, edges


class AcocMatrix:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.vertices, self.edges = init_matrix(self.x_size, self.y_size)

    def show_plot(self):
        x_coord, y_coord = zip(*[(v.x, v.y) for v in self.vertices])
        for edge in self.edges:
            plt.plot([edge.vertex_a.x, edge.vertex_b.x], [edge.vertex_a.y, edge.vertex_b.y], 'k-')
        plt.plot(x_coord, y_coord, 'o')
        plt.axis([min(x_coord) - 1, self.x_size, min(y_coord) - 1, self.y_size])
        plt.show()

    def find_vertex(self, x_y):
        for v in self.vertices:
            if (v.x, v.y) == x_y:
                return v
        return None
        # return next((v for v in self.vertices if v.coordinates() == x_y), None)


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

    def coordinates(self):
        return self.x, self.y


def main():
    matrix = AcocMatrix(5, 5)

    v = matrix.vertices[0]
    e = v.connected_edges[0]
    e.pheromone_strength = 0.2
    matrix.show_plot()

if __name__ == "__main__":
    main()
