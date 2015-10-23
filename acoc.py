#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import sys
import random
from itertools import repeat
from copy import copy

import numpy as np

from acoc_matrix import AcocMatrix
import acoc_plotter as plotter
from acoc_plotter import LivePheromonePlot
from ant import Ant
from is_point_inside import is_point_inside
import data_generator as dg


def normalize(values):
    """
    Normalizes a range of values from 0 to 1
    """
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)

    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def next_edge_and_vertex(matrix, ant):
    edges_travelled = ant.edges_travelled if len(ant.edges_travelled) > 0 else None

    v = matrix.find_vertex(ant.current_coordinates)
    if not v:
        pass
    connected_edges = copy(v.connected_edges)

    if edges_travelled is not None:
        for e in edges_travelled:
            if e in connected_edges:
                connected_edges.remove(e)
    if len(connected_edges) == 0:
        return None, None
    probabilities = normalize(np.array([e.pheromone_strength for e in connected_edges]))

    rand_num = random.random()
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


def cost_function(polygon, data):
    points = data.T.tolist()
    score = 0
    for p in points:
        if is_point_inside(p, polygon):
            score += 1 if p[2] == 0 else 0
        else:
            score += 1 if p[2] == 1 else 0
    return (score / data.shape[1]) / len(polygon)


def put_pheromones(path, data, pheromone_constant, pher_max):
    score = cost_function(path, data)

    unique_edges = get_unique_edges(path)
    for edge in unique_edges:
        new_pheromone_strength = edge.pheromone_strength + pheromone_constant * score
        edge.pheromone_strength = new_pheromone_strength if new_pheromone_strength < pher_max else pher_max


def pheromone_evaporation(matrix, pheromone_constant, evaporation_const):
    for edge in matrix.edges:
        rand_num = random.random()
        if rand_num < evaporation_const:
            edge.pheromone_strength = pheromone_constant


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def classify(data, ant_count, pheromone_constant, evaporation_const, pher_max, live_plot):
    ant_scores = []
    current_best_polygon = []
    current_best_score = 0

    matrix = AcocMatrix(data)

    if live_plot:
        live_plot = LivePheromonePlot(matrix, data)

    while len(ant_scores) < ant_count:
        start_vertex = matrix.vertices[random.randint(0, len(matrix.vertices) - 1)]
        start_coordinates = (start_vertex.x, start_vertex.y)
        ant = Ant(start_coordinates)

        edge, ant.current_coordinates = next_edge_and_vertex(matrix, ant)
        ant.edges_travelled.append(edge)
        ant_at_target = ant_is_stuck = False

        while not ant_at_target and not ant_is_stuck:
            if ant.current_coordinates == start_coordinates or len(ant.edges_travelled) > 10000:
                ant_at_target = True
            else:
                edge, ant.current_coordinates = next_edge_and_vertex(matrix, ant)
                if edge is None:
                    ant_is_stuck = True
                else:
                    ant.edges_travelled.append(edge)

        if ant_is_stuck:
            continue
        ant_score = cost_function(ant.edges_travelled, data)
        if ant_score > current_best_score:
            current_best_polygon = ant.edges_travelled
            current_best_score = ant_score

        put_pheromones(current_best_polygon, data, pheromone_constant, pher_max)
        pheromone_evaporation(matrix, 0.1, evaporation_const)

        ant_scores.append(ant_score)
        print_on_current_line("Ant: {}/{}".format(len(ant_scores) + 1, ant_count))
        if live_plot and len(ant_scores) % 50 == 0:
            live_plot.update(matrix.edges)

    if live_plot:
        live_plot.close()

    return ant_scores, current_best_polygon


def run(ant_count, iteration_count, pheromone_constant, evaporation_const, pher_max, live_plot=False):
    all_ant_scores = np.zeros((iteration_count, ant_count))
    global_best_polygon = list(repeat(0, 9999))
    global_best_score = 0

    red = np.insert(dg.uniform_rectangle((1, 3), (2, 4), 200), 2, 0, axis=0)
    blue = np.insert(dg.uniform_rectangle((4, 6), (2, 4), 200), 2, 1, axis=0)

    # Two-dimensional array with x-coordinates in first array and y-coordinates in second array
    data = np.concatenate((red, blue), axis=1)
    for i in range(iteration_count):
        print("\nIteration: {}/{}".format(i + 1, iteration_count))

        ant_scores, path = \
            classify(data, ant_count, pheromone_constant, evaporation_const, pher_max, live_plot)
        print_on_current_line("Best ant score: {}".format(max(ant_scores)))

        all_ant_scores[i, :] = ant_scores
        if max(ant_scores) > global_best_score:
            global_best_polygon = path
            global_best_score = max(ant_scores)

    print("\nGlobal best ant score: {}".format(global_best_score))

    plotter.plot_path_with_data(global_best_polygon, data)
    plotter.plot_ant_scores(ant_scores)


if __name__ == "__main__":
    run(100, 5, 1, 0.01, 5.0, False)
