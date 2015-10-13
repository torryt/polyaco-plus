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


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)
    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def remove_object_from_list(object_list, object):
    for i, e in enumerate(object_list):
        if e == object:
            object_list.pop(i)
            return True
    return False


def next_edge_and_vertex(matrix, ant):
    prev_edge = ant.edges_travelled[-1] if len(ant.edges_travelled) > 0 else None

    connected_edges = matrix.find_vertex(ant.current_coordinates).connected_edges
    if prev_edge:
        remove_object_from_list(connected_edges, prev_edge)

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

def put_pheromones(matrix, path, pheromone_constant):
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


def shortest_path(matrix, start_coord, target_coord, ant_count, pheromone_constant, decay_constant, live_plot):
    path_lengths = []
    current_shortest_path = list(repeat(0, 9999))

    if live_plot:
        live_plot = LivePheromonePlot(matrix, start_coord, target_coord)

    for i in range(ant_count):
        ant = Ant(start_coord)

        ant_at_target = False
        while not ant_at_target:
            if ant.current_coordinates != target_coord:
                edge, ant.current_coordinates = next_edge_and_vertex(matrix, ant)
                ant.edges_travelled.append(edge)
                pass
            else:
                ant_at_target = True
        if is_shorter_path(ant.edges_travelled, current_shortest_path):
            current_shortest_path = ant.edges_travelled
        put_pheromones(matrix, current_shortest_path, pheromone_constant)
        pheromones_decay(matrix, 0.1, decay_constant)

        path_lengths.append(len(ant.edges_travelled))
        print_on_current_line("Ant: {}/{}".format(i+1, ant_count))
        if live_plot and i % 20 == 0:

            live_plot.update(matrix.edges)

    if live_plot:
        live_plot.close()

    return path_lengths, current_shortest_path


def main():
    ant_count = 400
    iteration_count = 5
    pheromone_constant = 15.0
    decay_constant = 0.04

    all_path_lengths = np.zeros((iteration_count,ant_count))
    global_shortest_path = list(repeat(0, 9999))

    for i in range(iteration_count):
        print("\nIteration: {}/{}".format(i+1, iteration_count))
        mtrx = AcocMatrix(20, 20)
        path_lengths, s_path = \
            shortest_path(mtrx, (1, 1), (15, 15), ant_count, pheromone_constant, decay_constant, True)
        print_on_current_line("Shortest path length: {}".format(len(s_path)))

        all_path_lengths[i, :] = path_lengths
        if is_shorter_path(s_path, global_shortest_path):
            global_shortest_path = s_path

    plotter.draw_all(all_path_lengths.mean(0), global_shortest_path, mtrx)


if __name__ == "__main__":
    main()
