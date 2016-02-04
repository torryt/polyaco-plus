from itertools import product

import numpy as np
import math
from itertools import repeat

from acoc.edge import Edge
from acoc.vertex import Vertex


DIRECTION = {'RIGHT': 0, 'LEFT': 1, 'UP': 2, 'DOWN': 3}


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
        [connect_vertex_to_edges(v, self.edges) for v in self.vertices]

    def add_to_plot(self, ax):
        for edge in self.edges:
            ax.plot([edge.a.x, edge.b.x], [edge.a.y, edge.b.y], '--', color='#CFCFCF')
        for i, v in enumerate(self.vertices):
            if i % 2 == 0:
                ax.plot(v.x, v.y, 'o', color='w')
            else:
                ax.plot(v.x, v.y, 'o', color='k')
        ax.axis([self.x_min_max[0] - 1, self.x_min_max[1] + 1, self.y_min_max[0] - 1, self.y_min_max[1] + 1])

    def level_up(self, polygon=None):
        self.level += 1
        self.granularity = self.granularity * 2 - 1
        edge_y_length = (self.y_min_max[1] - self.y_min_max[0]) / (self.granularity - 1)
        edge_x_length = (self.x_min_max[1] - self.x_min_max[0]) / (self.granularity - 1)

        new_edges = []
        new_vertices = []

        for edge in self.edges:
            mid_v = Vertex(((edge.a.x + edge.b.x) / 2), ((edge.a.y + edge.b.y) / 2))
            self.vertices.append(mid_v)
            new_vertices.append(mid_v)

            new_edges.extend([Edge(edge.a, mid_v, edge.pheromone_strength),
                              Edge(mid_v, edge.b, edge.pheromone_strength)])
            if polygon:
                if edge in polygon:
                    polygon.remove(edge)
                    polygon.extend(new_edges[-2:])
        [connect_vertex_to_edges(v, new_edges) for v in new_vertices]
        self.edges = new_edges

        # horizontal_edges = filter(lambda e: e.a.y > [ed.a.y for ed in e.a.connected_edges],
        #                           filter(lambda e: e.a.y < self.y_min_max[1],
        #                                  filter(lambda e: math.isclose(e.a.y, e.b.y), self.edges)))

        def has_no_vertical_edges(v):
            return all(map(lambda e: math.isclose(e.a.y, e.b.y), v.connected_edges))
        vcs = filter(lambda v: has_no_vertical_edges(v) and v.y < self.y_min_max[1], new_vertices)

        for down in vcs:
            mid_v = Vertex(down.x, down.y + edge_y_length)
            right = down.connected_edges[DIRECTION['RIGHT']].b.connected_edges[DIRECTION['UP']].b
            left = down.connected_edges[DIRECTION['LEFT']].a.connected_edges[DIRECTION['UP']].b
            up = left.connected_edges[DIRECTION['UP']].b.connected_edges[DIRECTION['RIGHT']].b

            self.edges.extend([Edge(down, mid_v, pheromone_strength=self.tau_initial),
                               Edge(left, mid_v, pheromone_strength=self.tau_initial),
                               Edge(mid_v, right, pheromone_strength=self.tau_initial),
                               Edge(mid_v, up, pheromone_strength=self.tau_initial)])
            self.vertices.append(mid_v)
        [connect_vertex_to_edges(v, self.edges) for v in self.vertices]


def find_vertex(x, y, vertices):
    try:
        return filter(lambda v: v == Vertex(x, y), vertices).__next__()
    except StopIteration:
        return None


def connect_vertex_to_edges(vertex, edges):
    vertex.connected_edges = ce = [None] * 4
    for e in edges:
        if e.b.x > vertex.x:
            ce[DIRECTION['RIGHT']] = e
        elif e.a.x < vertex.x:
            ce[DIRECTION['LEFT']] = e
        elif e.b.y > vertex.y:
            ce[DIRECTION['UP']] = e
        elif e.a.y < vertex.y:
            ce[DIRECTION['DOWN']] = e


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
    from acoc.acoc_plotter import plot_pheromones, plot_matrix
    from utils import generate_folder_name

    save = False
    show = True
    plot = False
    data = np.array([[[0, 0]], [[1, 1]]])
    mtrx = AcocMatrix(data, granularity=3)
    pol = mtrx.edges[:4]
    mtrx.edges[1].pheromone_strength = 5
    mtrx.edges[0].pheromone_strength = 3
    save_folder = generate_folder_name()
    if plot:
        # plot_pheromones(mtrx, data, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
        plot_matrix(mtrx, show=show, save=save)
    for i in range(1):
        mtrx.level_up(pol)
        print("Edges: {}, vertices: {}".format(len(mtrx.edges), len(mtrx.vertices)))
        if plot:
            plot_matrix(mtrx, show=show, save=save)
            # plot_pheromones(mtrx, data, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
