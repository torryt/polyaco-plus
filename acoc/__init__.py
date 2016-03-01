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
from itertools import combinations

import acoc.acoc_plotter as plotter
import utils
from acoc import ray_cast
from acoc.acoc_matrix import AcocMatrix
from acoc.ant import Ant
from acoc.polygon import polygon_to_array, polygon_length
from acoc.ray_cast import is_point_inside
from utils import normalize
from config import CLASSIFIER_CONFIG

odd = np.vectorize(ray_cast.odd)


class PolyACO:
    def __init__(self, dimensions, config=None, save_folder=None):
        self.config = config if config is not None else CLASSIFIER_CONFIG
        self.save_folder = save_folder

        self.planes = list(combinations(range(dimensions), 2))
        self.polygons = None

    def evaluate(self, test_data):
        if self.polygons is None:
            raise RuntimeError('PolyACO must be trained before evaluation')
        plane_scores = []
        for plane, poly in zip(self.planes, self.polygons):
            plane_data = np.take(test_data, list(plane), axis=1)
            plane_scores.append(ray_cast.is_points_inside_cuda(plane_data, polygon_to_array(poly)))
        aggregated_scores = np.mean(np.array(plane_scores), axis=0)
        predictions = [0 if v > 0.5 else 1 for v in list(aggregated_scores)]
        return predictions

    def train(self, training_data, target):
        self.polygons = []
        for i, plane in enumerate(self.planes):
            plane_string = str(plane[0]) + str(plane[1])
            plane_data = np.concatenate((np.take(training_data, list(plane), axis=1), np.array([target]).T), axis=1)
            _polygon = self._train_plane(plane_data, plane_string, print_string=', Plane {}/{}'.format(i + 1, len(self.planes)))
            self.polygons.append(_polygon)

    def _train_plane(self, data, plane_string, print_string=''):
        cuda.to_device(data)
        ant_scores = []
        current_best_ant = []
        last_level_up_or_best_ant = 0
        self.matrix = AcocMatrix(data, tau_initial=self.config.tau_init)

        current_best_score = 0
        best_ant_history = [None] * self.config.run_time
        best_ant_history[0] = current_best_score

        t_start = process_time()
        t_elapsed = 0

        def plot_pheromones():
            if self.config.plot:
                plotter.plot_pheromones(self.matrix, data.T, self.config.tau_min, self.config.tau_max, file_name='ant' + str(len(ant_scores)),
                                        save=True, folder_name=osp.join(self.save_folder, 'pheromones/'), title="Ant {}".format(len(ant_scores)))

        def print_status():
            while t_elapsed < self.config.run_time:
                if self.config.multi_level:
                    to_print = "Ant: {}, Time elapsed: {:.1f} seconds, Level {}".format(
                        len(ant_scores), process_time() - t_start, self.matrix.level) + print_string
                else:
                    to_print = "Ant: {}, Time elapsed: {:.1f} seconds".format(
                        len(ant_scores), process_time() - t_start) + print_string
                utils.print_on_current_line(to_print)
                time.sleep(0.1)

        def update_history():
            while t_elapsed < self.config.run_time:
                best_ant_history[int(t_elapsed)] = current_best_score
                time.sleep(1)
        Thread(target=print_status).start()
        Thread(target=update_history).start()

        while t_elapsed < self.config.run_time:
            start_vertex = get_random_weighted(self.matrix.edges)
            if self.config.multi_level:
                if (len(ant_scores) - last_level_up_or_best_ant) > self.config.convergence_rate:
                    if self.config.max_level is None or self.matrix.level < self.config.max_level:
                        # plot_pheromones()
                        self.matrix.level_up(current_best_ant)
                        last_level_up_or_best_ant = len(ant_scores)
            _ant = Ant(start_vertex)
            _ant.move_ant()

            while not _ant.at_target and not _ant.is_stuck:
                _ant.move_ant()
            if _ant.at_target:
                ant_score = self._score(_ant.edges_travelled, data)
                if ant_score > current_best_score:
                    current_best_ant = _ant.edges_travelled
                    current_best_score = ant_score
                    last_level_up_or_best_ant = len(ant_scores)
                    if self.config.plot:
                        plotter.plot_path_with_data(current_best_ant, data, self.matrix, save=True,
                                                    save_folder=osp.join(self.save_folder, 'best_paths/{}/'.format(plane_string)),
                                                    file_name='ant' + str(len(ant_scores)))
                        # plot_pheromones()

                self._put_pheromones(current_best_ant, current_best_score)
                self._reset_at_random(self.matrix)
                ant_scores.append(ant_score)
                t_elapsed = process_time() - t_start

        for i, e in enumerate(best_ant_history):
            if e is None:
                best_ant_history[i] = next(_e for _e in reversed(best_ant_history[:i]) if _e is not None)
        return current_best_ant

    def _reset_at_random(self, matrix):
        for edge in matrix.edges:
            rand_num = random.random()
            if rand_num < self.config.rho:
                edge.pheromone_strength = self.config.tau_min

    def _grad_pheromone_decay(self, matrix):
        for edge in matrix.edges:
            edge.pheromone_strength *= 1-self.config.rho

    def _score(self, polygon, data):
        edges = polygon_to_array(polygon)
        if self.config.gpu:
            cost = cost_function_gpu(data, edges)
        else:
            cost = cost_function(data, edges)
        # try:
        length_factor = 1 / polygon_length(polygon)
        return (cost**self.config.alpha) * (length_factor**self.config.beta)

    def _put_pheromones(self, path, score):
        for edge in path:
            pheromone_strength = edge.pheromone_strength + score
            edge.pheromone_strength = pheromone_strength if pheromone_strength < self.config.tau_max else self.config.tau_max


def get_random_weighted(edges):
    weights = normalize(np.array([e.pheromone_strength for e in edges]))
    random_weighted_edge = np_choice(edges, p=weights)
    return random.choice([random_weighted_edge.a, random_weighted_edge.b])


def cost_function(points, edges):
    is_inside = np.array([ray_cast.is_point_inside(p, edges) for p in points])
    score = np.sum(np.logical_xor(is_inside, points[:, 2]))
    return score / points.shape[0]


def cost_function_gpu(points, edges):
    is_inside = ray_cast.is_points_inside_cuda(points, edges)
    score = np.sum(np.logical_xor(is_inside, points[:, 2]))
    return score / points.shape[0]