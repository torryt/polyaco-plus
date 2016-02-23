#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import os.path as osp
import random
import time
from threading import Thread
from time import process_time

import numpy as np
from numba import cuda, jit
from numpy.random.mtrand import choice as np_choice

import acoc.acoc_plotter as plotter
import utils
from acoc import ray_cast
from acoc.acoc_matrix import AcocMatrix
from acoc.ant import Ant
from acoc.polygon import polygon_to_array, polygon_length
from acoc.ray_cast import is_point_inside
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


def cost_function(points, edges):
    is_inside = np.array([ray_cast.is_point_inside(p, edges) for p in points])
    score = np.sum(np.logical_xor(is_inside, points[:, 2]))
    return score / points.shape[0]


def cost_function_gpu(points, edges):
    is_inside = ray_cast.is_points_inside_cuda(points, edges)
    score = np.sum(np.logical_xor(is_inside, points[:, 2]))
    return score / points.shape[0]


@jit
def cost_function_jit(points, edges):
    is_inside = np.empty((points.shape[0]))
    for i in range(points.shape[0]):
        is_inside[i] = ray_cast.is_point_inside_jit(points[i], edges)
    score = np.sum(np.logical_xor(is_inside, points[:, 2]))
    return score / points.shape[0]


class Classifier:
    def __init__(self, config, save_folder=''):
        self.config = config
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
        self.multi_level = config['multi_level']

        self.granularity = config['granularity']
        self.nest_grid = config['nest_grid']
        self.max_level = config['max_level']
        if self.multi_level:
            self.convergence_rate = config['convergence_rate']
            self.granularity = 3
        self.gpu = config['gpu']
        self.matrix = None

    def classify(self, data, plot=False, print_string=''):
        cuda.to_device(data.T)
        ant_scores = []
        current_best_polygon = []
        last_level_up_or_best_ant = 0
        self.matrix = AcocMatrix(data,
                                 tau_initial=self.tau_init,
                                 granularity=self.granularity,
                                 nest_grid=self.nest_grid,
                                 max_level=self.max_level)
        plotter.plot_matrix_and_data(self.matrix, data, show=True)

        current_best_score = 0
        best_ant_history = [None] * self.run_time
        best_ant_history[0] = current_best_score

        t_start = process_time()
        t_elapsed = 0

        def plot_pheromones():
            if plot:
                plotter.plot_pheromones(self.matrix, data, self.tau_min, self.tau_max, file_name='ant' + str(len(ant_scores)),
                                        save=True, folder_name=osp.join(self.save_folder, 'pheromones/'))

        def print_status():
            while t_elapsed < self.run_time:
                if self.multi_level:
                    to_print = "Ant: {}, Time elapsed: {:.1f} seconds, Level {}".format(
                        len(ant_scores), process_time() - t_start, self.matrix.level) + print_string
                else:
                    to_print = "Ant: {}, Time elapsed: {:.1f} seconds".format(
                        len(ant_scores), process_time() - t_start) + print_string
                utils.print_on_current_line(to_print)
                time.sleep(0.1)

        def update_history():
            while t_elapsed < self.run_time:
                best_ant_history[int(t_elapsed)] = current_best_score
                time.sleep(1)
        Thread(target=print_status).start()
        Thread(target=update_history).start()

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
                    if self.max_level is None or self.matrix.level < self.max_level:
                        plot_pheromones()
                        self.matrix.level_up(current_best_polygon)
                        current_best_score = self.score(current_best_polygon, data)
                        last_level_up_or_best_ant = len(ant_scores)

            _ant = Ant(start_vertex)
            _ant.move_ant()

            while not _ant.at_target and not _ant.is_stuck:
                _ant.move_ant()
            if _ant.at_target:
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

        for i, e in enumerate(best_ant_history):
            if e is None:
                best_ant_history[i] = next(_e for _e in reversed(best_ant_history[:i]) if _e is not None)
        return best_ant_history, current_best_polygon,

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
            cost = cost_function(data.T, edges)
        # try:
        length_factor = 1 / polygon_length(polygon)
        # Handles very rare and weird error where length of polygon == 0
        # except ZeroDivisionError:
        #     length_factor = 1
        return (cost**self.alpha) * (length_factor**self.beta)

    def put_pheromones(self, path, data, score):
        for edge in path:
            pheromone_strength = edge.pheromone_strength + score
            edge.pheromone_strength = pheromone_strength if pheromone_strength < self.tau_max else self.tau_max
