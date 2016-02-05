#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import random
import numpy as np
from numpy.random.mtrand import choice as np_choice
import math
from numba import cuda
import os.path as osp
import time
from time import process_time
from threading import Thread

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
    random_weighted_edge = np_choice(edges, p=weights)
    return random.choice([random_weighted_edge.a, random_weighted_edge.b])


def select_from_global_best(matrix, current_best_polygon):
    if len(current_best_polygon) != 0:
        select_edge = random.choice(current_best_polygon)
        return random.choice([select_edge.a, select_edge.b])
    else:
        return matrix.vertices[random.randint(0, len(matrix.vertices) - 1)]


def select_with_chance_of_global_best(matrix, current_best_polygon):
    if len(current_best_polygon) != 0:
        # 50% probability for selecting a point from current best path
        if random.randint(0, 1) == 0:
            select_edge = random.choice(current_best_polygon)
            return select_edge.a
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
    return np.array([[[e.a.x, e.a.y], [e.b.x, e.b.y]] for e in polygon], dtype='float32')


def polygon_length(polygon):
    x_edge = next(e for e in polygon if e.a.y == e.b.y)
    x_edge_length = x_edge.b.x - x_edge.a.x
    x_edges_count = len([1 for e in polygon if e.a.y == e.b.y])

    y_edge = next(e for e in polygon if e.a.x == e.b.x)
    y_edge_length = y_edge.b.y - y_edge.a.y
    y_edges_count = len(polygon) - x_edges_count
    return (x_edges_count * x_edge_length) + (y_edges_count * y_edge_length)


class Classifier:
    def __init__(self, config, save_folder=''):
        self.run_time = config['run_time']
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
        self.multi_level = config['multi_level']
        if self.multi_level:
            self.max_level = config['max_level']
            self.convergence_rate = config['convergence_rate']

        self.gpu = config['gpu']
        self.matrix = None

    def classify(self, data, plot=False, print_string=''):
        ant_scores = []
        current_best_polygon = []
        last_level_up_or_best_ant = 0
        current_best_score = 0
        self.matrix = AcocMatrix(data, tau_initial=self.tau_init, granularity=self.granularity)
        best_ant_history = np.empty(int(self.run_time))

        t_start = process_time()
        t_elapsed = 0

        def plot_pheromones():
            if plot:
                plotter.plot_pheromones(self.matrix, data, self.tau_min, self.tau_max, file_name='ant' + str(len(ant_scores)),
                                        save=True, folder_name=osp.join(self.save_folder, 'pheromones/'))

        def print_status():
            while t_elapsed < self.run_time:
                utils.print_on_current_line(
                    "Ant: {}, Time elapsed: {:.1f} seconds".format(
                        len(ant_scores), process_time() - t_start) + print_string)
                time.sleep(0.1)

        def log_best_ants():
            while t_elapsed < self.run_time:
                best_ant_history[int(t_elapsed)] = current_best_score
                time.sleep(1)
        Thread(target=print_status).start()
        Thread(target=log_best_ants).start()

        while t_elapsed < self.run_time:
            if self.ant_init == 'static':
                start_vertex = self.matrix.vertices[0]
            elif self.ant_init == 'weighted':
                start_vertex = get_random_weighted(self.matrix.edges)
            elif self.ant_init == 'on_global_best':
                start_vertex = select_from_global_best(self.matrix, current_best_polygon)
            elif self.ant_init == 'chance_of_global_best':
                start_vertex = select_with_chance_of_global_best(self.matrix, current_best_polygon)
            else:  # Random
                start_vertex = self.matrix.vertices[random.randint(0, len(self.matrix.vertices) - 1)]
            if self.multi_level:
                if (len(ant_scores) - last_level_up_or_best_ant) > self.convergence_rate:
                    plot_pheromones()
                    # Cannot have more than three levels because Matrix is a trilogy
                    if self.matrix.level <= self.max_level:
                        print("\nLeveling up to level {}. Granularity is now {}".
                              format(self.matrix.level, self.matrix.granularity * 2 - 1))
                        self.matrix.level_up(current_best_polygon)
                        current_best_score = self.score(current_best_polygon, data)
                        last_level_up_or_best_ant = len(ant_scores)
                        plot_pheromones()
                    else:
                        is_converged = False
                        print("\nConverged. Is exploding! Run away!!!")
                        break

            _ant = Ant(start_vertex)
            _ant.move_ant()

            while not _ant.at_target and not _ant.is_stuck:
                _ant.move_ant()

            ant_score = self.score(_ant.edges_travelled, data)
            if ant_score > current_best_score:
                current_best_polygon = _ant.edges_travelled
                current_best_score = ant_score
                last_level_up_or_best_ant = len(ant_scores)
                if plot:
                    plot_pheromones()
                    plotter.plot_path_with_data(current_best_polygon, data, self.matrix, save=True,
                                                save_folder=osp.join(self.save_folder, 'best_paths/'),
                                                file_name='ant' + str(len(ant_scores)))

            self.put_pheromones(current_best_polygon, data, current_best_score)
            if self.decay_type == 'probabilistic':
                self.reset_at_random(self.matrix)
            elif self.decay_type == 'gradual':
                self.grad_pheromone_decay(self.matrix)
            ant_scores.append(ant_score)
            t_elapsed = process_time() - t_start

        return ant_scores, current_best_polygon, best_ant_history

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
            length_factor = 1 / polygon_length(polygon)
        # Handles very rare and weird error where length of polygon == 0
        except ZeroDivisionError:
            length_factor = 1
        return (cost**self.alpha) * (length_factor**self.beta)

    def put_pheromones(self, path, data, score):
        for edge in path:
            mtx_edge = self.matrix.edges[self.matrix.edges.index(edge)]
            pheromone_strength = mtx_edge.pheromone_strength + score
            mtx_edge.pheromone_strength = pheromone_strength if pheromone_strength < self.tau_max else self.tau_max
