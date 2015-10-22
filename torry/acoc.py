#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import random
from itertools import repeat
from copy import copy

import numpy as np

from torry.acoc_matrix import AcocMatrix
import torry.acoc_plotter as plotter
from torry.acoc_plotter import LivePheromonePlot
from torry.ant import Ant
from torry.is_point_inside import is_point_inside
import torry.data_generator as dg


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)

    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def next_edge_and_vertex(matrix, ant):
    prev_edge = ant.edges_travelled[-1] if len(ant.edges_travelled) > 0 else None

    v = matrix.find_vertex(ant.current_coordinates)
    if not v:
        pass
    connected_edges = copy(v.connected_edges)

    if prev_edge:
        if prev_edge in connected_edges:
            connected_edges.remove(prev_edge)
    probabilities = normalize_0_to_1(np.array([e.pheromone_strength for e in connected_edges]))

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


def classification_score(polygon, data):
    points = data.T.tolist()

    reward = 0
    for p in points:
        if is_point_inside(p, polygon):
            reward += 1 if p[2] == 0 else -1
    return reward


def put_pheromones(path, data, pheromone_constant):
    reward = classification_score(path, data)

    unique_edges = get_unique_edges(path)
    for edge in unique_edges:
        if reward > 0:
            edge.pheromone_strength += pheromone_constant * reward


def pheromones_decay(matrix, pheromone_constant, decay_constant):
    for edge in matrix.edges:
        rand_num = random.random()
        if rand_num < decay_constant:
            edge.pheromone_strength = pheromone_constant


def print_on_current_line(in_string):
    out_string = "\r" + in_string
    sys.stdout.write(out_string)
    sys.stdout.flush()


def classify(data, ant_count, pheromone_constant, decay_constant, live_plot):
    ant_scores = []
    current_best_polygon = []
    current_best_score = 0

    matrix = AcocMatrix(data)

    if live_plot:
        live_plot = LivePheromonePlot(matrix, data)

    for i in range(ant_count):
        start_vertex = matrix.vertices[random.randint(0, len(matrix.vertices) - 1)]
        start_coordinates = (start_vertex.x, start_vertex.y)
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

        ant_score = classification_score(ant.edges_travelled, data)
        if ant_score > current_best_score:
            current_best_polygon = ant.edges_travelled
            current_best_score = ant_score

        put_pheromones(current_best_polygon, data, pheromone_constant)
        pheromones_decay(matrix, 0.1, decay_constant)

        ant_scores.append(ant_score)
        print_on_current_line("Ant: {}/{}".format(i + 1, ant_count))
        if live_plot and i % 50 == 0:
            live_plot.update(matrix.edges)

    if live_plot:
        live_plot.close()

    return ant_scores, current_best_polygon


def run(ant_count, iteration_count, pheromone_constant, decay_constant, live_plot=False):
    all_ant_scores = np.zeros((iteration_count, ant_count))
    global_best_polygon = list(repeat(0, 9999))
    global_best_score = 0

    red = np.insert(dg.uniform_rectangle((2, 4), (2, 4), 20), 2, 0, axis=0)
    blue = np.insert(dg.uniform_rectangle((6, 8), (2, 4), 20), 2, 1, axis=0)

    # Two-dimensional array with x-coordinates in first array and y-coordinates in second array
    data = np.concatenate((red, blue), axis=1)
    # plotter.plot_data(data)
    for i in range(iteration_count):
        print("\nIteration: {}/{}".format(i + 1, iteration_count))

        ant_scores, path = \
            classify(data, ant_count, pheromone_constant, decay_constant, live_plot)
        print_on_current_line("Best ant score: {}".format(max(ant_scores)))

        all_ant_scores[i, :] = ant_scores
        if max(ant_scores) > global_best_score:
            global_best_polygon = path

    print("\nGlobal best ant score: {}".format(global_best_score))
    plotter.draw_all(all_ant_scores.mean(0), global_best_polygon, data)
    plotter.plot_path_with_data(global_best_polygon, data)


if __name__ == "__main__":
    run(100, 1, 0.5, 0.1, True)
