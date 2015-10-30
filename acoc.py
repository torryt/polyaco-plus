#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
from copy import copy
import numpy as np
from numpy.random import choice

from acoc_matrix import AcocMatrix
from acoc_plotter import LivePheromonePlot
from ant import Ant
from is_point_inside import is_point_inside
import utils


def normalize(values):
    """
    Normalizes a range of values to values from 0 to 1
    """
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)

    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def next_edge_and_vertex(matrix, ant):
    edges_travelled = ant.edges_travelled if len(ant.edges_travelled) > 0 else None
    connected_edges = copy(matrix.find_vertex(ant.current_coordinates).connected_edges)

    if edges_travelled is not None:
        for e in edges_travelled:
            if e in connected_edges:
                connected_edges.remove(e)
    if len(connected_edges) == 0:
        return None, None

    weights = normalize(np.array([e.pheromone_strength for e in connected_edges]))
    selected = choice(connected_edges, p=weights)
    if selected.vertex_a.coordinates() != ant.current_coordinates:
        return selected, selected.vertex_a.coordinates()
    else:
        return selected, selected.vertex_b.coordinates()


def get_unique_edges(path):
    z = set(path)
    unique_edges = list(z)
    return unique_edges


def polygon_score(polygon, data):
    points = data.T.tolist()
    score = 0
    for p in points:
        if is_point_inside(p, polygon):
            score += 1 if p[2] == 0 else 0
        else:
            score += 1 if p[2] == 1 else 0
    return score / data.shape[1]


class Classifier:
    def __init__(self, ant_count, q, q_min, q_max, rho, alpha, beta):
        self.ant_count = ant_count
        self.q = q
        self.q_min = q_min
        self.q_max = q_max
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

    def classify(self, data, live_plot):
        ant_scores = []
        current_best_polygon = []
        current_best_score = 0

        matrix = AcocMatrix(data, q_initial=self.q_min)

        if live_plot:
            live_plot = LivePheromonePlot(matrix, data)

        while len(ant_scores) < self.ant_count:
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
            ant_score = self.cost_function(ant.edges_travelled, data)
            if ant_score > current_best_score:
                current_best_polygon = ant.edges_travelled
                current_best_score = ant_score

            self.put_pheromones(current_best_polygon, data)
            self.reset_at_random(matrix)

            ant_scores.append(ant_score)
            if live_plot and len(ant_scores) % 20 == 0:
                live_plot.update(matrix.edges)
            utils.print_on_current_line("Ant: {}/{}".format(len(ant_scores), self.ant_count))

        if live_plot:
            live_plot.close()

        return ant_scores, current_best_polygon

    def reset_at_random(self, matrix):
        for edge in matrix.edges:
            rand_num = random.random()
            if rand_num < self.rho:
                edge.pheromone_strength = self.q_min

    def cost_function(self, polygon, data):
        score = polygon_score(polygon, data)
        try:
            length_factor = 1/len(polygon)
        except ZeroDivisionError:
            length_factor = self.beta
        return (score**self.alpha) * (length_factor**self.beta)

    def put_pheromones(self, path, data):
        score = self.cost_function(path, data)

        unique_edges = get_unique_edges(path)
        for edge in unique_edges:
            new_pheromone_strength = edge.pheromone_strength + (self.q * score)
            edge.pheromone_strength = new_pheromone_strength if new_pheromone_strength < self.q_max else self.q_max
