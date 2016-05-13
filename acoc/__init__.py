#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import os.path as osp
import random
import time
from datetime import datetime
from warnings import warn

import numpy as np
from numba import cuda, jit
from numpy.random.mtrand import choice as np_choice
from itertools import combinations
from copy import copy

import acoc.acoc_plotter as plotter
import utils
from acoc import ray_cast
from acoc.ray_cast import is_points_inside_cuda
from acoc.acoc_matrix import AcocMatrix
from acoc.ant import Ant
from acoc.polygon import polygon_to_array, polygon_length
from acoc.ray_cast import is_point_inside
from utils import normalize
from config import CLASSIFIER_CONFIG

odd = np.vectorize(ray_cast.odd)


class PolyACO:
    def __init__(self, dimensions, class_indices, config=None, save_folder=datetime.utcnow().strftime('%Y-%m-%d_%H%M')):
        self.config = config if config is not None else CLASSIFIER_CONFIG
        self.save_folder = save_folder

        if len(class_indices) > len(set(class_indices)):
            warn("Class index array contains duplicate entries")
        self.class_indices = np.unique(class_indices)
        self.planes = list(combinations(range(dimensions), 2))
        self.model = []

    def evaluate(self, test_data):
        if len(self.model) == 0:
            raise RuntimeError('PolyACO must be trained before evaluation')
        inside = np.empty((len(self.planes), len(self.class_indices), test_data.shape[0]), dtype=bool)
        for i, plane in enumerate(self.model):
            plane_data = np.take(test_data, list(self.planes[i]), axis=1)
            for j, poly in enumerate(plane):
                inside[i, j, :] = is_points_inside_cuda(plane_data, polygon_to_array(poly))

        aggregated_scores = np.mean(inside, axis=0)
        predictions = np.empty((test_data.shape[0]), dtype=int)
        max_elements = np.argmax(aggregated_scores, axis=0)
        for i in range(test_data.shape[0]):
            predictions[i] = self.class_indices[max_elements[i]]
        return predictions

    def train(self, training_data, target, start_time=time.time()):
        for i, plane_axes in enumerate(self.planes):
            plane = []
            self.model.append(plane)
            num_class = len(self.class_indices)
            for j in range(num_class):
                j_class = self.class_indices[j]
                plane_string = "plane{}{}_class{}".format(plane_axes[0], plane_axes[1], j_class)

                new_t = copy(target)
                new_t[new_t != j_class] = -1
                new_t[new_t == j_class] = 0
                new_t[new_t == -1] = 1
                plane_data = np.concatenate((np.take(training_data, list(plane_axes), axis=1), np.array([new_t]).T), axis=1)
                print_string = "Polygon {}/{}".format(i * len(self.class_indices) + (j + 1),
                                                len(self.planes) * len(self.class_indices))
                _polygon = self._construct_polygon(plane_data, plane_string, start_time, print_string)
                plane.append(_polygon)
            p_data = np.append(np.take(training_data, list(self.planes[i]), axis=1).T, [target], axis=0).T
            if self.config.save:
                fn = "plane_dims_{}_{}".format(plane_axes[0], plane_axes[1])
                plotter.plot_paths_with_data(self.model[i], p_data,
                                             save_folder=osp.join(self.save_folder, 'planes'), file_name=fn)

    def _construct_polygon(self, data, plane_string, start_time, print_string):
        cuda.to_device(data)
        ant_scores = []
        current_best_ant = []
        last_level_up_or_best_ant = 0
        matrix = AcocMatrix(data, tau_initial=self.config.tau_init)
        current_best_score = 0

        while matrix.level <= self.config.max_level:
            start_vertex = get_random_weighted(matrix.edges)
            if self.config.multi_level:
                if (len(ant_scores) - last_level_up_or_best_ant) > self.config.level_convergence_rate:
                    matrix.level_up(current_best_ant)
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
                        plotter.plot_path_with_data(current_best_ant, data, matrix, save=True,
                                                    save_folder=osp.join(self.save_folder,
                                                                         'best_paths/{}/'.format(plane_string)),
                                                    file_name='ant' + str(len(ant_scores)))

                self._put_pheromones(current_best_ant, current_best_score)
                self._reset_at_random(matrix)
                ant_scores.append(ant_score)
                t_elapsed = utils.seconds_to_hms(time.time() - start_time)
                utils.print_on_current_line("Level {}/{}, {}, Time elapsed: {}".format(matrix.level, self.config.max_level, print_string, t_elapsed))

        return current_best_ant

    def _reset_at_random(self, matrix):
        for edge in matrix.edges:
            rand_num = random.random()
            if rand_num < self.config.rho:
                edge.pheromone_strength = self.config.tau_min

    def _score(self, polygon, data):
        edges = polygon_to_array(polygon)
        if self.config.gpu:
            cost = cost_function_gpu(data, edges)
        else:
            cost = cost_function(data, edges)
        # try:
        length_factor = 1 / polygon_length(polygon)
        return (cost ** self.config.alpha) * (length_factor ** self.config.beta)

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


def compute_score(predictions, ground_truth):
    return np.equal(predictions, ground_truth).mean() * 100
