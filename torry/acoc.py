#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from random import random
from itertools import repeat

import numpy as np

from acoc_matrix import AcocMatrix
import acoc_plotter as plt
from acoc_plotter import LivePheromonePlot
from ant import Ant


ant_count = 100
# TODO: Prøv å øke denne
pheromone_constant = 20.0
decay_constant = 0.005

# TODO Alltid styrke pheromonene til den beste løsningen


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)
    normalize_const = 1.0 / values.sum()
    return values * normalize_const


# TODO Se om det er no bugs her
def next_edge_and_vertex(matrix, ant):
    ignore_edges = [ant.edges_travelled[-1]] if len(ant.edges_travelled) > 0 else []
    connected_edges = matrix.get_connected_edges(ant.current_vertex, ignore_edges)
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
    plt.plot_path(global_shortest_path, matrix)

    print("\nShortest path length: {}".format(len(global_shortest_path)))
    return results


if __name__ == "__main__":
    # TODO Ta gjennomsnittet av mange kjøringer, f.eks 100
    mtrx = AcocMatrix(20, 20)
    ant_p, shortest_path = shortest_path(mtrx, (1, 1), (15, 15))
    plt.plot_pheromone_values(mtrx)
    plt.plot_path_lengths(ant_p)
