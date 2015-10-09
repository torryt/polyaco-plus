#!/usr/bin/env python
# -*- coding: utf-8 -*-
from acoc_matrix import AcocMatrix
from random import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat

# TODO: Prøv med en maur
ant_count = 10
iterations = 1000
# TODO: Prøv å øke denne
pheromone_constant = 1.0
decay_constant = 0.1


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)
    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def next_edge(matrix, ant):
    # TODO: Uten ignore_list
    connected_edges = \
        matrix.get_connected_edges(ant.current_vertex, ant.edges_travelled)
    if len(connected_edges) == 0:
        return ant.current_vertex, False
    probabilities = normalize_0_to_1(np.array([e.pheromone_strength for e in connected_edges]))

    # TODO: Random are the same for all in iteration
    p = random()
    cumulative_prob = 0
    for i, edge in enumerate(connected_edges):
        cumulative_prob += probabilities[i]
        if p <= cumulative_prob:
            if edge.a_vertex != ant.current_vertex:
                return edge, edge.a_vertex,
            else:
                return edge, edge.b_vertex


def all_has_completed_tour(ants, target_vertex):
    for ant in ants:
        if ant.current_vertex:
            last_edge = ant.edges_travelled[-1]
            if not last_edge.has_vertex(target_vertex):
                return False
    return True


# TODO: Sett pheromone for hver unike edge
def put_pheromones(matrix, path, target_vertex):
    for edge in path:
        for orig_edge in matrix.edges:
            if orig_edge.has_both_vertices(edge.a_vertex, edge.b_vertex):
                orig_edge.pheromone_strength += pheromone_constant / len(path)


def pheromones_decay(matrix):
    # TODO: Slett pheromones med en viss sannsynlighet
    for edge in matrix.edges:
        edge.pheromone_strength *= (1 - pheromone_constant)


def iteration_result(matrix, ants):
    return np.array([len(ant.edges_travelled) for ant in ants]).mean()


def show_plot_results(results):
    x_coord = range(0, len(results))
    y_coord = results
    plt.plot(x_coord, y_coord)
    plt.axis([0, max(x_coord), 0, max(y_coord)])
    plt.show()


def has_shorter_path(ants, global_shortest_path):
    for ant in ants:
        if len(ant.edges_travelled) < len(global_shortest_path):
            global_shortest_path = ant.edges_travelled
            return True, global_shortest_path
    return False, global_shortest_path


def remove_dead_ended_ants(ants):
    for ant in ants:
        if not ant.current_vertex:
            ants.remove(ant)
    return ants


def shortest_path(matrix, start_vertex, target_vertex):
    results = []
    global_shortest_path = list(repeat(0, 2000))

    for iteration in range(iterations):
        ants = [Ant(start_vertex)] * ant_count

        all_ants_at_target = False
        while not all_ants_at_target:
            for ant in ants:
                if not ant.current_vertex == target_vertex and ant.current_vertex:
                    edge, ant.current_vertex = next_edge(matrix, ant)
                    ant.edges_travelled.append(edge)
            all_ants_at_target = all_has_completed_tour(ants, target_vertex)

        ants = remove_dead_ended_ants(ants)
        is_shorter_path = has_shorter_path(ants, global_shortest_path)
        if is_shorter_path[0]:
            global_shortest_path = is_shorter_path[1]
            put_pheromones(matrix, global_shortest_path, target_vertex)

        # pheromones_decay(matrix)
        iter_result = iteration_result(matrix, ants)
        results.append(iter_result)
        if iteration % 1 == 0:
            print("Iteration {} avg. path length: {}".format(iteration, iter_result))
    print("Shortest path length: {}".format(min(results)))
    show_plot_results(results)
    return results


class Ant:
    def __init__(self, start_vertex):
        self.current_vertex = start_vertex
        self.edges_travelled = []

if __name__ == "__main__":
    shortest_path(AcocMatrix(10, 10), (1, 1), (8, 8))
