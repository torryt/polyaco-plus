#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import random
from copy import copy
import numpy as np
from numpy.random.mtrand import choice

import utils
from acoc.acoc_matrix import AcocMatrix
from acoc.acoc_plotter import LivePheromonePlot
import acoc.acoc_plotter as plotter
from acoc.ant import Ant
from utils import normalize
from acoc.is_point_inside import is_point_inside
from utils import normalize


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


class Classifier:
    def __init__(self, config, save_folder=''):
        self.ant_count = config['ant_count']
        self.q = config['q']
        self.q_min = config['q_min']
        self.q_max = config['q_max']
        self.q_init = config['q_init']  
        self.rho = config['rho']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.ant_init = config['ant_init']
        self.decay_type = config['decay_type']
        self.save_folder = save_folder

    def classify(self, data, plot=False, print_string=''):
        ant_scores = []
        current_best_polygon = []
        current_best_score = 0
        matrix = AcocMatrix(data, q_initial=self.q_init)

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
            ant_score = self.cost_function(_ant.edges_travelled, data)
            if ant_score > current_best_score:
                current_best_polygon = _ant.edges_travelled
                current_best_score = ant_score

            self.put_pheromones(current_best_polygon, data)
            if self.decay_type == 'random_type':
                self.reset_at_random(matrix)
            elif self.decay_type == 'grad_type':
                self.grad_pheromone_decay(matrix)

            ant_scores.append(ant_score)

            if plot and len(ant_scores) % 50 == 0:
                plotter.plot_pheromones(matrix, data, self.q_min, self.q_max, True, self.save_folder)
                # live_plot.update(matrix.edges)
            utils.print_on_current_line("Ant: {}/{}".format(len(ant_scores), self.ant_count) + print_string)

        # if live_plot:
        #     live_plot.close()

        return ant_scores, current_best_polygon

    def reset_at_random(self, matrix):
        for edge in matrix.edges:
            rand_num = random.random()
            if rand_num < self.rho:
                edge.pheromone_strength = self.q_min

    def grad_pheromone_decay(self, matrix):
        for edge in matrix.edges:
            edge.pheromone_strength *= 1-self.rho

    def polygon_score(self, polygon, data):
        points = data.T.tolist()
        score = 0
        unique_polygon = copy(polygon)
        for vertex in unique_polygon:
            if vertex.twin in unique_polygon:
                unique_polygon.remove(vertex.twin)
        for vertex in points:
            if is_point_inside(vertex, unique_polygon):
                score += 1 if vertex[2] == 0 else 0
            else:
                score += 1 if vertex[2] != 0 else 0
        return score / data.shape[1]

    def cost_function(self, polygon, data):
        score = self.polygon_score(polygon, data)
        try:
            length_factor = 1/len(polygon)
        # Handles very rare and weird error
        except ZeroDivisionError:
            length_factor = self.beta
        return (score**self.alpha) * (length_factor**self.beta)

    def put_pheromones(self, path, data):
        score = self.cost_function(path, data)

        unique_edges = get_unique_edges(path)
        for edge in unique_edges:
            new_pheromone_strength = edge.pheromone_strength + (self.q * score)
            edge.pheromone_strength = new_pheromone_strength if new_pheromone_strength < self.q_max else self.q_max
