from itertools import product

import numpy as np
import math
from itertools import repeat

from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.ray_cast import any_point_inside


DIRECTION = {'RIGHT': 0, 'LEFT': 1, 'UP': 2, 'DOWN': 3}


class AcocMatrix:
    def __init__(self, data, tau_initial=1.0, granularity=10, old_matrix=None):
        self.granularity = granularity
        self.x_min_max = np.amin(data[0]) - .1, np.amax(data[0]) + .1
        self.y_min_max = np.amin(data[1]) - .1, np.amax(data[1]) + .1
        self.init_edge_length_x = (self.x_min_max[1] - self.x_min_max[0]) / (self.granularity - 1)
        self.init_edge_length_y = (self.y_min_max[1] - self.y_min_max[0]) / (self.granularity - 1)

        self.tau_initial = tau_initial
        self.level = 0

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=2, endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=2, endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = [Vertex(x, y) for x, y in coordinates]

        # The number of edges |E| depends on number of vertices with the following formula |E| = y(x-1) + x(y-1)
        # Example: A 4x4 matrix will generate 4(4-1) + 4(4-1) = 24
        self.edges = create_edges(self.vertices, self.tau_initial)

        increase_section_granularity(data, self.edges)
        connect_vertices_to_edges(self.edges)


def increase_section_granularity(data, section):
    if any_point_inside(data, section):
        pass


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

    save = False
    show = False
    plot = False
    dt = np.array([[[0, 0]], [[1, 1]]])
    mtrx = AcocMatrix(dt, granularity=3)
    pol = mtrx.edges[:4]
    mtrx.edges[1].pheromone_strength = 5
    mtrx.edges[0].pheromone_strength = 3
    save_folder = generate_folder_name()
    print("Level {}: Granularity {}, edges {}, vertices {}".format(mtrx.level, mtrx.granularity, len(mtrx.edges), len(mtrx.vertices)))

    if plot:
        plot_pheromones(mtrx, dt, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
        plot_matrix(mtrx, show=show, save=save)
    for i in range(6):
        mtrx.level_up(pol)
        print("Level {}: Granularity {}, edges {}, vertices {}".format(mtrx.level, mtrx.granularity, len(mtrx.edges), len(mtrx.vertices)))
        if plot:
            plot_matrix(mtrx, show=show, save=save)
            plot_pheromones(mtrx, dt, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
