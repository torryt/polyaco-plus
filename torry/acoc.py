#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from random import random
from itertools import repeat

import numpy as np

from acoc_matrix import AcocMatrix
import acoc_plotter as plotter
from acoc_plotter import LivePheromonePlot
from ant import Ant


ant_count = 400
iteration_count = 10
pheromone_constant = 15.0
decay_constant = 0.04


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)
    normalize_const = 1.0 / values.sum()
    return values * normalize_const


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


def get_unique_edges(path):
    z = set(path)
    unique_edges = list(z)
    return unique_edges


def put_pheromones(matrix, path):
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


def is_shorter_path(path_a, path_b):
    if len(path_a) < len(path_b):
        return True
    return False


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def shortest_path(matrix, start_vertex, target_vertex, live_plot=True):
    path_lengths = []
    current_shortest_path = list(repeat(0, 9999))

    if live_plot:
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
        if is_shorter_path(ant.edges_travelled, current_shortest_path):
            current_shortest_path = ant.edges_travelled
        put_pheromones(matrix, current_shortest_path)
        pheromones_decay(matrix, 0.1)

        path_lengths.append(len(ant.edges_travelled))
        print_on_current_line("Ant: {}/{}".format(i+1, ant_count))
        if live_plot and i % 40 == 0:
            live_plot.update(matrix.edges)

    if live_plot:
        live_plot.close()

    return path_lengths, current_shortest_path


def main():
    all_path_lengths = np.zeros((iteration_count,ant_count))
    global_shortest_path = list(repeat(0, 9999))

    for i in range(iteration_count):
        print("\nIteration: {}/{}".format(i+1, iteration_count))
        mtrx = AcocMatrix(20, 20)
        path_lengths, s_path = shortest_path(mtrx, (1, 1), (15, 15), False)
        print_on_current_line("Shortest path length: {}".format(len(s_path)))

        all_path_lengths[i, :] = path_lengths
        if is_shorter_path(s_path, global_shortest_path):
            global_shortest_path = s_path

    # TODO Slå sammen disse plottene til en figur
    plotter.plot_pheromone_values(mtrx)
    plotter.plot_path(global_shortest_path, mtrx)
    plotter.plot_path_lengths(all_path_lengths.mean(0))


if __name__ == "__main__":
    main()
