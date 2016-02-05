from itertools import product

import numpy as np
import math
from itertools import repeat

from acoc.edge import Edge
from acoc.vertex import Vertex


DIRECTION = {'RIGHT': 0, 'LEFT': 1, 'UP': 2, 'DOWN': 3}


class AcocMatrix:
    def __init__(self, data, tau_initial=1.0, granularity=10, old_matrix=None):
        self.granularity = granularity
        self.x_min_max = np.amin(data[0]) - .1, np.amax(data[0]) + .1
        self.y_min_max = np.amin(data[1]) - .1, np.amax(data[1]) + .1
        self.edge_length_x = (self.x_min_max[1] - self.x_min_max[0]) / (self.granularity - 1)
        self.edge_length_y = (self.y_min_max[1] - self.y_min_max[0]) / (self.granularity - 1)

        self.tau_initial = tau_initial
        self.level = 1

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=int(self.granularity), endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=int(self.granularity), endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = [Vertex(x, y) for x, y in coordinates]

        # The number of edges |E| depends on number of vertices with the following formula |E| = y(x-1) + x(y-1)
        # Example: A 4x4 matrix will generate 4(4-1) + 4(4-1) = 24
        self.edges = create_edges(self.vertices, self.tau_initial)
        connect_vertices_to_edges(self.edges)

    def level_up(self, polygon=None):
        self.level += 1
        self.granularity = self.granularity * 2 - 1
        self.edge_length_x = (self.x_min_max[1] - self.x_min_max[0]) / (self.granularity - 1)
        self.edge_length_y = (self.y_min_max[1] - self.y_min_max[0]) / (self.granularity - 1)

        new_edges = []
        new_vertices = []

        for edge in self.edges:
            v_mid = Vertex(((edge.a.x + edge.b.x) / 2), ((edge.a.y + edge.b.y) / 2))
            self.vertices.append(v_mid)
            new_vertices.append(v_mid)

            new_edges.extend([Edge(edge.a, v_mid, edge.pheromone_strength),
                              Edge(v_mid, edge.b, edge.pheromone_strength)])
            if polygon:
                if edge in polygon:
                    polygon.remove(edge)
                    polygon.extend(new_edges[-2:])
        self.edges = new_edges
        connect_vertices_to_edges(self.edges)

        def has_no_vertical_edges(v):
            return v.connected_edges[DIRECTION['UP']] is None and \
                   v.connected_edges[DIRECTION['DOWN']] is None
        vcs = filter(lambda v: has_no_vertical_edges(v) and v.y < self.y_min_max[1], new_vertices)
        for v_down in vcs:
            v_right = v_down.connected_edges[DIRECTION['RIGHT']].b.connected_edges[DIRECTION['UP']].b
            v_left = v_down.connected_edges[DIRECTION['LEFT']].a.connected_edges[DIRECTION['UP']].b
            v_up = v_left.connected_edges[DIRECTION['UP']].b.connected_edges[DIRECTION['RIGHT']].b
            v_mid = Vertex(v_down.x, (v_up.y + v_down.y) / 2)

            self.edges.extend([Edge(v_down, v_mid, pheromone_strength=self.tau_initial),
                               Edge(v_left, v_mid, pheromone_strength=self.tau_initial),
                               Edge(v_mid, v_right, pheromone_strength=self.tau_initial),
                               Edge(v_mid, v_up, pheromone_strength=self.tau_initial)])
            self.vertices.append(v_mid)
        connect_vertices_to_edges(self.edges)


def find_vertex(x, y, vertices):
    try:
        return filter(lambda v: v == Vertex(x, y), vertices).__next__()
    except StopIteration:
        return None


def connect_vertices_to_edges(edges):
    for e in edges:
        # if horizontal edge
        if e.a.y == e.b.y:
            e.a.connected_edges[DIRECTION['RIGHT']] = e
            e.b.connected_edges[DIRECTION['LEFT']] = e
        else:
            e.a.connected_edges[DIRECTION['UP']] = e
            e.b.connected_edges[DIRECTION['DOWN']] = e


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

    save = True
    show = False
    plot = False
    dt = np.array([[[0, 0]], [[1, 1]]])
    mtrx = AcocMatrix(dt, granularity=3)
    pol = mtrx.edges[:4]
    mtrx.edges[1].pheromone_strength = 5
    mtrx.edges[0].pheromone_strength = 3
    save_folder = generate_folder_name()
    if plot:
        plot_pheromones(mtrx, dt, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
        plot_matrix(mtrx, show=show, save=save)
    for i in range(5):
        mtrx.level_up(pol)
        print("Level {}: Granularity {}, edges {}, vertices {}".format(i + 1, mtrx.granularity, len(mtrx.edges), len(mtrx.vertices)))
        if plot:
            plot_matrix(mtrx, show=show, save=save)
            plot_pheromones(mtrx, dt, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
