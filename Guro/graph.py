__author__ = 'Guro'

from itertools import product
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.vertices = self.get_vertices(x_size, y_size)
        self.edges = self.get_edges(x_size, y_size)

    def get_vertices(self, x_size,y_size):
        return list(product(range(1,x_size + 1), range(1, y_size + 1)))

    def get_edges(self, x_size, y_size):
        return self._generate_edges(x_size, y_size)

    def _generate_edges(self, x_size, y_size):
        edges = []

        for x ,y in self.vertices:
            if not x == x_size:
                east_neighbor = (x + 1, y)
                edges.append(GraphEdge((x, y), east_neighbor))
            if not y == y_size:
                north_neighbor = (x, y + 1)
                edges.append(GraphEdge((x ,y), north_neighbor))
        return edges

    def find_connected_edges(self,vertex):
        edges = []
        for edge in self.edges:
            if edge.vertex1 == vertex or edge.vertex2 == vertex:
                edges.append(edge)
        return edges

    def draw_plot(self):
        x_coord, y_coord = zip(*self.vertices)
        for edge in self.edges:
            plt.plot([edge.vertex1[0], edge.vertex2[0]], [edge.vertex1[1], edge.vertex2[1]], 'k-')
        plt.plot(x_coord, y_coord, 'o')
        plt.axis([0, self.x_size + 1, 0, self.y_size + 1])
        plt.show()

class GraphEdge:
    def __init__(self, vertex1, vertex2, pheromone_strength=0.1):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.pheromone_strength = pheromone_strength

    def __repr__(self):
        return str((self.vertex1, self.vertex2, self.pheromone_strength))

if __name__ =="__main__":
    graph = Graph(10,10)
    graph.draw_plot()
