from itertools import product

import numpy as np
from copy import copy

from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.ray_cast import points_of_both_classes_inside, points_inside
from acoc.polygon import polygon_to_array
from utils import data_manager

DIRECTION = {'RIGHT': 0, 'LEFT': 1, 'UP': 2, 'DOWN': 3}


class AcocMatrix:
    def __init__(self, data, tau_initial=1.0):
        self.data = data
        self.x_min_max = np.amin(data[:, 0]) - .1, np.amax(data[:, 0]) + .1
        self.y_min_max = np.amin(data[:, 1]) - .1, np.amax(data[:, 1]) + .1
        self.init_edge_length_x = (self.x_min_max[1] - self.x_min_max[0])
        self.init_edge_length_y = (self.y_min_max[1] - self.y_min_max[0])
        self.tau_initial = tau_initial
        self.sections = []
        self.level = 0

        x_coord = np.linspace(self.x_min_max[0], self.x_min_max[1], num=2, endpoint=True)
        y_coord = np.linspace(self.y_min_max[0], self.y_min_max[1], num=2, endpoint=True)

        coordinates = list(product(x_coord, y_coord))
        self.vertices = [Vertex(x, y) for x, y in coordinates]

        # The number of edges |E| depends on number of vertices with the following formula |E| = y(x-1) + x(y-1)
        # Example: A 4x4 matrix will generate 4(4-1) + 4(4-1) = 24
        self.edges = create_edges(self.vertices, self.tau_initial)
        connect_edges_to_vertices(self.edges)
        self.level_up()

    def not_leveled_sections(self):
        if self.level == 0:
            return [copy(self.edges)]
        else:
            return self.sections

    def level_up(self, best_polygon=None):
        sections = self.not_leveled_sections()
        new_sections, new_edges, remove_edges = ([] for _ in range(3))

        for s in sections:
            result = self.increase_section_granularity(s, best_polygon)
            if result is not None:
                new_sections.extend(result[0])
                new_edges.extend(result[1])
                remove_edges.extend(result[2])

        self.edges = list((set(new_edges) | set(self.edges)) - set(remove_edges))
        self.sections = new_sections
        self.level += 1
        connect_edges_to_vertices(self.edges)

    def increase_section_granularity(self, section, best_polygon=None):
        p_inside = points_inside(self.data, polygon_to_array(section))

        if not all_points_have_same_position(p_inside) and have_points_of_both_classes(p_inside):
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
                if best_polygon:
                    if edge in best_polygon:
                        best_polygon.remove(edge)
                        best_polygon.extend(new_edges[-2:])

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
            return subsections, new_edges, remove_edges


def all_points_have_same_position(points):
    coords = points[:, :2]
    for i in range(points.shape[0]):
        if not np.allclose(coords[0], coords[i]):
            return False
    return True


def have_points_of_both_classes(points):
    args_class = points[:, 2]
    if 0 in args_class and 1 in args_class:
        return True
    return False


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
    from acoc.acoc_plotter import plot_pheromones, plot_matrix_and_data
    from utils import generate_folder_name

    plot = True
    save = True
    show = False
    data_set = data_manager.load_data_set('breast_cancer')

    data = data_set.data[:, :2]
    target = np.array([data_set.target]).T
    np.place(target, target == 4, 1)
    np.place(target, target == 2, 0)
    dt = np.concatenate((data, target), axis=1)

    # dt = np.array([[0, 0.02, 0],
    #                [0.05, 0, 1],
    #                [1, 1, 1]])
    mtrx = AcocMatrix(dt)
    pol = mtrx.edges[:4]
    mtrx.edges[1].pheromone_strength = 5
    mtrx.edges[0].pheromone_strength = 3
    save_folder = generate_folder_name()
    print("Level {}: Edges {}, vertices {}".format(mtrx.level, len(mtrx.edges), len(mtrx.vertices)))

    if plot:
        # plot_pheromones(mtrx, dt, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
        plot_matrix_and_data(mtrx, dt, show=show)
    for i in range(6):
        mtrx.level_up(pol)
        print("Level {}: Edges {}, vertices {}".format(mtrx.level, len(mtrx.edges), len(mtrx.vertices)))
        if plot:
            plot_matrix_and_data(mtrx, dt, show=show, save=save)
            # plot_pheromones(mtrx, dt, tau_min=1, tau_max=10, folder_name=save_folder, save=save, show=show)
