from itertools import product

import numpy as np
import math
from itertools import repeat
from copy import copy

import acoc.acoc_plotter as plotter
from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.ray_cast import any_point_inside
from acoc.polygon import polygon_to_array


DIRECTION = {'RIGHT': 0, 'LEFT': 1, 'UP': 2, 'DOWN': 3}


class AcocMatrix:
    def __init__(self, data, tau_initial=1.0, max_level=3):
        self.data = data
        self.x_min_max = np.amin(data[0]) - .1, np.amax(data[0]) + .1
        self.y_min_max = np.amin(data[1]) - .1, np.amax(data[1]) + .1
        self.init_edge_length_x = (self.x_min_max[1] - self.x_min_max[0])
        self.init_edge_length_y = (self.y_min_max[1] - self.y_min_max[0])

        self.tau_initial = tau_initial
        self.max_level = max_level

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=2, endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=2, endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = [Vertex(x, y) for x, y in coordinates]

        # The number of edges |E| depends on number of vertices with the following formula |E| = y(x-1) + x(y-1)
        # Example: A 4x4 matrix will generate 4(4-1) + 4(4-1) = 24
        self.edges = create_edges(self.vertices, self.tau_initial)
        new_edges, remove_edges = self.increase_section_granularity(copy(self.edges), 0)
        self.edges = list(set(new_edges) - set(remove_edges))
        connect_edges_to_vertices(self.edges)

    def increase_section_granularity(self, section, level):
        if level == self.max_level:
            return

        if any_point_inside(self.data, polygon_to_array(section)):
            new_edges = []
            remove_edges = []

            # Cut edges in two
            for edge in section:
                v_mid = Vertex(((edge.a.x + edge.b.x) / 2), ((edge.a.y + edge.b.y) / 2))
                if v_mid not in self.vertices:
                    self.vertices.append(v_mid)
                else:
                    v_mid = next(v for v in self.vertices if v == v_mid)
                new_edges.extend([Edge(edge.a, v_mid, edge.pheromone_strength),
                                  Edge(v_mid, edge.b, edge.pheromone_strength)])

            remove_edges.extend(section)
            connect_edges_to_vertices(new_edges)
            v_lower_left = section[0].a
            v_down = v_lower_left.connected_edges[DIRECTION['RIGHT']].b
            v_right = v_down.connected_edges[DIRECTION['RIGHT']].b.connected_edges[DIRECTION['UP']].b
            v_left = v_lower_left.connected_edges[DIRECTION['UP']].b
            v_up = v_left.connected_edges[DIRECTION['UP']].b.connected_edges[DIRECTION['RIGHT']].b
            v_center = Vertex(v_down.x, v_left.y)

            new_edges.extend([Edge(v_down, v_center, pheromone_strength=self.tau_initial),
                               Edge(v_left, v_center, pheromone_strength=self.tau_initial),
                               Edge(v_center, v_right, pheromone_strength=self.tau_initial),
                               Edge(v_center, v_up, pheromone_strength=self.tau_initial)])
            self.vertices.append(v_center)
            connect_edges_to_vertices(new_edges)

            corners = [v_lower_left,
                       v_down,
                       v_left,
                       v_center]

            subsections = []
            for c in corners:
                subsections.append([
                    c.connected_edges[DIRECTION['RIGHT']],
                    c.connected_edges[DIRECTION['RIGHT']].b.connected_edges[DIRECTION['UP']],
                    c.connected_edges[DIRECTION['UP']].b.connected_edges[DIRECTION['RIGHT']],
                    c.connected_edges[DIRECTION['UP']]
                ])

            for sbs in subsections:
                result = self.increase_section_granularity(sbs, level+1)
                if result is not None:
                    new_edges.extend(result[0])
                    remove_edges.extend(result[1])
            return new_edges, remove_edges


def connect_edges_to_vertices(edges):
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
