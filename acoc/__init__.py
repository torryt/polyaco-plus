#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import random
from copy import copy
import numpy as np
from numpy.random.mtrand import choice
import math

import utils
from acoc.acoc_matrix import AcocMatrix
import acoc.acoc_plotter as plotter
from acoc.ant import Ant
from acoc.ray_cast import is_point_inside
from acoc import ray_cast
from utils import normalize

odd = np.vectorize(ray_cast.odd)


def get_unique_edges(path):
    z = set(path)
    unique_edges = list(z)
    return unique_edges


def get_random_weighted(edges):
    weights = normalize(np.array([e.pheromone_strength for e in edges]))
    random_weighted_edge = choice(edges, p=weights)
    return random_weighted_edge.start


def get_global(matrix, current_best_polygon):
    if len(current_best_polygon) != 0:
        select_edge = random.choice(current_best_polygon)
        return select_edge.start
    else:
        return matrix.vertices[random.randint(0, len(matrix.vertices) - 1)]


def get_chance_of_global(matrix, current_best_polygon):
    if len(current_best_polygon) != 0:
        # 50% probability for selecting a point from current best path
        if random.randint(0, 1) == 0:
            select_edge = random.choice(current_best_polygon)
            return select_edge.start
        else:
            return matrix.vertices[random.randint(0, len(matrix.vertices) - 1)]
    else:
        return matrix.vertices[random.randint(0, len(matrix.vertices) - 1)]


def cost_function_gpu(edges, points):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(edges.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(points.shape[0] / threads_per_block[0])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    result = np.empty((edges.shape[0], points.shape[0]), dtype=bool)
    ray_cast.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](edges, points, result)

    is_inside = odd(np.sum(result, axis=0))
    score = np.logical_xor(is_inside, points[:, 2])

    return np.sum(score) / points.shape[0]


def cost_function(polygon, data):
    points = data.T.tolist()
    unique_polygon = copy(polygon)
    score = 0
    for vertex in unique_polygon:
        if vertex.twin in unique_polygon:
            unique_polygon.remove(vertex.twin)
    for vertex in points:
        if is_point_inside(vertex, unique_polygon):
            score += 1 if vertex[2] == 0 else 0
        else:
            score += 1 if vertex[2] != 0 else 0
    return score / data.shape[1]


class Classifier:
    def __init__(self, config, save_folder=''):
        self.ant_count = config['ant_count']
        self.tau_min = config['tau_min']
        self.tau_max = config['tau_max']
        self.tau_init = config['tau_init']
        self.rho = config['rho']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.ant_init = config['ant_init']
        self.decay_type = config['decay_type']
        self.save_folder = save_folder
        self.granularity = config['granularity']
        self.gpu = config['gpu']

    def classify(self, data, plot=False, print_string=''):
        ant_scores = []

        current_best_polygon = []
        current_best_score = 0
        matrix = AcocMatrix(data, tau_initial=self.tau_init, granularity=self.granularity)

        while len(ant_scores) < self.ant_count:
            if self.ant_init == 'static':
                start_vertex = matrix.vertices[0]
            elif self.ant_init == 'weighted':
                start_vertex = get_random_weighted(matrix.edges)
            elif self.ant_init == 'on_global_best':
                start_vertex = get_global(matrix, current_best_polygon)
            elif self.ant_init == 'chance_of_global_best':
                start_vertex = get_chance_of_global(matrix, current_best_polygon)
            else:  # Random
                start_vertex = matrix.vertices[random.randint(0, len(matrix.vertices) - 1)]

            _ant = Ant(start_vertex)
            edge = _ant.move_ant()
            _ant.edges_travelled.append(edge)
            ant_at_target = ant_is_stuck = False

            while not ant_at_target and not ant_is_stuck:
                if _ant.current_edge.target == start_vertex or len(_ant.edges_travelled) > 10000:
                    ant_at_target = True
                else:
                    edge = _ant.move_ant()
                    if edge is None:
                        ant_is_stuck = True
                    else:
                        _ant.edges_travelled.append(edge)

            if ant_is_stuck:
                continue
            ant_score, ant_cost = self.score(_ant.edges_travelled, data)
            if ant_score > current_best_score:
                current_best_polygon = _ant.edges_travelled
                current_best_score = ant_score

            self.put_pheromones(current_best_polygon, data, current_best_score)
            if self.decay_type == 'probabilistic':
                self.reset_at_random(matrix)
            elif self.decay_type == 'gradual':
                self.grad_pheromone_decay(matrix)

            ant_scores.append(ant_score)

            if plot and len(ant_scores) % 200 == 0:
                plotter.plot_pheromones(matrix, data, self.tau_min, self.tau_max, True, self.save_folder)
            utils.print_on_current_line("Ant: {}/{}".format(len(ant_scores), self.ant_count) + print_string)

        return ant_scores, current_best_polygon

    def reset_at_random(self, matrix):
        for edge in matrix.edges:
            rand_num = random.random()
            if rand_num < self.rho:
                edge.pheromone_strength = self.tau_min

    def grad_pheromone_decay(self, matrix):
        for edge in matrix.edges:
            edge.pheromone_strength *= 1-self.rho

    def score(self, polygon, data):
        if self.gpu:
            twins_removed = copy(polygon)
            for vertex in twins_removed:
                if vertex.twin in twins_removed:
                    twins_removed.remove(vertex.twin)

            edges = np.array([[[e.start.x, e.start.y], [e.target.x, e.target.y]] for e in twins_removed], dtype='float32')
            cost = cost_function_gpu(edges, data.T)
        else:
            cost = cost_function(polygon, data)
        try:
            length_factor = 1/len(polygon)
        # Handles very rare and weird error where length of polygon == 0
        except ZeroDivisionError:
            length_factor = 1
        return (cost**self.alpha) * (length_factor**self.beta), cost

    def put_pheromones(self, path, data, score):
        unique_edges = get_unique_edges(path)
        for edge in unique_edges:
            pheromone_strength = edge.pheromone_strength + score
            edge.pheromone_strength = pheromone_strength if pheromone_strength < self.tau_max else self.tau_max