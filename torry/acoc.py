#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from random import random
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt

from acoc_matrix import AcocMatrix
from pheromone_plot import LivePheromonePlot

ant_count = 100
# TODO: Prøv å øke denne
pheromone_constant = 40.0
decay_constant = 0.005


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)
    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def next_edge_and_vertex(matrix, ant):
    ignore_edges = [ant.edges_travelled[-1]] if len(ant.edges_travelled)>0 else []
    connected_edges = \
        matrix.get_connected_edges(ant.current_vertex, ignore_edges)
    probabilities = normalize_0_to_1(np.array([e.pheromone_strength for e in connected_edges]))

    rand_num = random()
    cumulative_prob = 0
    for i, edge in enumerate(connected_edges):
        cumulative_prob += probabilities[i]
        if rand_num <= cumulative_prob:
            if edge.a_vertex != ant.current_vertex:
                return edge, edge.a_vertex
            else:
                return edge, edge.b_vertex


def all_has_completed_tour(ants, target_vertex):
    for ant in ants:
        if ant.current_vertex:
            last_edge = ant.edges_travelled[-1]
            if not last_edge.has_vertex(target_vertex):
                return False
    return True


def get_unique_edges(path):
    z = set(path)
    unique_edges = list(z)
    return unique_edges


def put_pheromones(matrix, path, target_vertex):
    unique_edges = get_unique_edges(path)
    for edge in unique_edges:
        for orig_edge in matrix.edges:
            if orig_edge.has_both_vertices(edge.a_vertex, edge.b_vertex):
                orig_edge.pheromone_strength += pheromone_constant / len(path)
                break


def pheromones_decay(matrix, initial_pheromone_value):
    for edge in matrix.edges:
        rand_num = random()
        if rand_num < decay_constant:
            edge.pheromone_strength = initial_pheromone_value


def show_plot_results(results):
    x_coordinates = range(0, len(results))
    y_coordinates = results
    plt.plot(x_coordinates, y_coordinates)
    plt.axis([0, max(x_coordinates), 0, max(y_coordinates)])
    plt.show()


def is_shorter_path(edges_travelled, global_shortest_path):
    if len(edges_travelled) < len(global_shortest_path):
        global_shortest_path = edges_travelled
        return True, global_shortest_path
    return False, global_shortest_path


def shortest_path(matrix, start_vertex, target_vertex):
    results = []
    global_shortest_path = list(repeat(0, 9999))

    live_plot = LivePheromonePlot(matrix, start_vertex, target_vertex)

    for i in range(ant_count):
        ant = Ant(start_vertex)

        ant_at_target = False
        while not ant_at_target:
            if ant.current_vertex != target_vertex:
                edge, ant.current_vertex = next_edge_and_vertex(matrix, ant)
                ant.edges_travelled.append(edge)
            else:
                ant_at_target = True
        shorter_path, global_shortest_path = is_shorter_path(ant.edges_travelled, global_shortest_path)
        if shorter_path:
            put_pheromones(matrix, global_shortest_path, target_vertex)

        results.append(ant.edges_travelled)
        pheromones_decay(matrix, 0.1)

        progress_string = "\rProgress: {}/{}".format(i, ant_count)
        sys.stdout.write(progress_string)
        sys.stdout.flush()
        live_plot.update(matrix.edges)

    live_plot.close()

    print("\nShortest path length: {}".format(len(global_shortest_path)))
    return results


class Ant:
    def __init__(self, start_vertex):
        self.current_vertex = start_vertex
        self.edges_travelled = []


def plot_path_lengths(ant_paths):
    x_coord = range(len(ant_paths))
    path_lengths = [len(p) for p in ant_paths]
    y_coord = path_lengths
    plt.plot(x_coord, y_coord, 'k-')
    plt.axis([0, len(ant_paths), 0, max(path_lengths)])
    plt.show()


def plot_pheromone_values(matrix):
    for edge in matrix.edges:
        line = plt.plot([edge.a_vertex[0], edge.b_vertex[0]], [edge.a_vertex[1], edge.b_vertex[1]], 'k-')
        plt.setp(line, linewidth=edge.pheromone_strength)
    plt.axis([-1, matrix.x_size, -1, matrix.y_size])
    plt.show()


if __name__ == "__main__":
    matrix = AcocMatrix(10, 10)
    ap = shortest_path(matrix, (1, 1), (8, 8))
    plot_pheromone_values(matrix)
    plot_path_lengths(ap)
