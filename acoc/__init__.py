#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import random
from copy import copy
import numpy as np
from numpy.random.mtrand import choice
import math
from numba import cuda

import utils
import acoc.acoc_plotter as plotter
from acoc.acoc_matrix import AcocMatrix
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


def cost_function_gpu(points, edges):
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(points.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(edges.shape[0] / threads_per_block[0])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    result = np.empty((points.shape[0], edges.shape[0]), dtype=bool)

    p_points = cuda.to_device(points)
    p_edges = cuda.to_device(edges)
    ray_cast.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](p_points, p_edges, result)

    is_inside = odd(np.sum(result, axis=1))
    score = np.sum(np.logical_xor(is_inside, points[:, 2]))
    return score / points.shape[0]


def cost_function_cpu(points, edges):
    is_inside = np.array([ray_cast.is_point_inside(p, edges) for p in points])
    score = np.sum(np.logical_xor(is_inside, points[:, 2]))
    return score / points.shape[0]


def polygon_to_array(polygon):
    twins_removed = copy(polygon)
    for edge in twins_removed:
        if edge.twin in twins_removed:
            twins_removed.remove(edge.twin)
    return np.array([[[e.start.x, e.start.y], [e.target.x, e.target.y]] for e in twins_removed], dtype='float32')


def remove_twins(polygon):
    twins_removed = copy(polygon)
    for edge in twins_removed:
        if edge.twin in twins_removed:
            twins_removed.remove(edge.twin)
    return twins_removed


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
        self.granularity = config['granularity']

    def classify(self, data, plot=False, print_string=''):
        ant_scores = []
        dropped = 0
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
                dropped += 1
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

            if plot and len(ant_scores) % 50 == 0:
                plotter.plot_pheromones(matrix, data, self.tau_min, self.tau_max, True, self.save_folder)

            utils.print_on_current_line("Ant: {}/{}".format(len(ant_scores), self.ant_count) + print_string)

        return ant_scores, current_best_polygon, dropped

    def reset_at_random(self, matrix):
        for edge in matrix.edges:
            rand_num = random.random()
            if rand_num < self.rho:
                edge.pheromone_strength = self.tau_min

    def grad_pheromone_decay(self, matrix):
        for edge in matrix.edges:
            edge.pheromone_strength *= 1-self.rho

    def score(self, polygon, data):
        edges = polygon_to_array(polygon)
        if self.gpu:
            cost = cost_function_gpu(data.T, edges)
        else:
            cost = cost_function_cpu(data.T, edges)
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
