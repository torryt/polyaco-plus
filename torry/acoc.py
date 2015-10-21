#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from random import random
from itertools import repeat
import numpy as np
from copy import copy

from torry.acoc_matrix import AcocMatrix
import torry.acoc_plotter as plotter
from torry.acoc_plotter import LivePheromonePlot
from torry.ant import Ant


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)

    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def next_edge_and_vertex(matrix, ant):
    prev_edge = ant.edges_travelled[-1] if len(ant.edges_travelled) > 0 else None

    connected_edges = copy(matrix.find_vertex(ant.current_coordinates).connected_edges)
    if prev_edge:
        if prev_edge in connected_edges:
            connected_edges.remove(prev_edge)
    probabilities = normalize_0_to_1(np.array([e.pheromone_strength for e in connected_edges]))

    rand_num = random()
    cumulative_prob = 0
    for i, edge in enumerate(connected_edges):
        cumulative_prob += probabilities[i]
        if rand_num <= cumulative_prob:
            if edge.vertex_a.coordinates() != ant.current_coordinates:
                return edge, edge.vertex_a.coordinates()
            else:
                return edge, edge.vertex_b.coordinates()
    return None


def get_unique_edges(path):
    z = set(path)
    unique_edges = list(z)
    return unique_edges


def put_pheromones(path, pheromone_constant):
    unique_edges = get_unique_edges(path)
    for edge in unique_edges:
        edge.pheromone_strength += pheromone_constant / len(path)


def pheromones_decay(matrix, pheromone_constant, decay_constant):
    for edge in matrix.edges:
        rand_num = random()
        if rand_num < decay_constant:
            edge.pheromone_strength = pheromone_constant


def is_shorter_path(path_a, path_b):
    if len(path_a) < len(path_b):
        return True
    return False


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def classify(data, ant_count, pheromone_constant, decay_constant, live_plot):
    path_lengths = []
    current_shortest_path = list(repeat(0, 9999))

    matrix_x_min_max = np.amin(data[0]) - 1, np.amax(data[0]) + 1
    matrix_y_min_max = np.amin(data[1]) - 1, np.amax(data[1]) + 1

    matrix = AcocMatrix(matrix_x_min_max, matrix_y_min_max)

    if live_plot:
        live_plot = LivePheromonePlot(matrix)

    for i in range(ant_count):
        start_coordinates = (1, 1)
        ant = Ant(start_coordinates)

        edge, ant.current_coordinates = next_edge_and_vertex(matrix, ant)
        ant.edges_travelled.append(edge)
        ant_at_target = False

        while not ant_at_target:
            if ant.current_coordinates == start_coordinates or len(ant.edges_travelled) > 10000:
                ant_at_target = True
            else:
                edge, ant.current_coordinates = next_edge_and_vertex(matrix, ant)
                ant.edges_travelled.append(edge)
        if is_shorter_path(ant.edges_travelled, current_shortest_path):
            current_shortest_path = ant.edges_travelled
        put_pheromones(current_shortest_path, pheromone_constant)
        pheromones_decay(matrix, 0.1, decay_constant)

        path_lengths.append(len(ant.edges_travelled))
        print_on_current_line("Ant: {}/{}".format(i + 1, ant_count))
        if live_plot and i % 50 == 0:
            live_plot.update(matrix.edges)
            pass

    if live_plot:
        live_plot.close()

    return path_lengths, current_shortest_path


def run(ant_count, iteration_count, pheromone_constant, decay_constant, live_plot=False):
    all_path_lengths = np.zeros((iteration_count, ant_count))
    global_shortest_path = list(repeat(0, 9999))

    # Two-dimensional array with x-coordinates in first array, and y-coordinates in second array
    data = np.array([[0, 10], [0, 10]])

    for i in range(iteration_count):
        print("\nIteration: {}/{}".format(i + 1, iteration_count))

        path_lengths, s_path = \
            classify(data, ant_count, pheromone_constant, decay_constant, live_plot)
        print_on_current_line("Shortest path length: {}".format(len(s_path)))

        all_path_lengths[i, :] = path_lengths
        if len(s_path) < len(global_shortest_path):
            global_shortest_path = s_path

    print("\nGlobal shortest path length: {}".format(len(global_shortest_path)))
    plotter.draw_all(all_path_lengths.mean(0), global_shortest_path, data)
    plotter.plot_path_with_data(global_shortest_path, data)


if __name__ == "__main__":
    run(400, 20, 5.0, 0.1, True)
