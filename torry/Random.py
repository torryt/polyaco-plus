import sys
from random import random
from itertools import repeat
import numpy as np

from torry.acoc_plotter import LivePheromonePlot
from torry.ant import Ant
from torry.acoc_matrix import AcocMatrix
import torry.acoc_plotter as plotter

ant_count = 100
iteration_count = 1


def equal_pheromones(values):
    return [1.0 / len(values)] * len(values)


def next_edge_and_vertex(matrix, ant):
    ignore_edges = [ant.edges_travelled[-1]] if len(ant.edges_travelled) > 0 else []
    connected_edges = matrix.get_connected_edges(ant.current_vertex, ignore_edges)
    probabilities = equal_pheromones(np.array([e.pheromone_strength for e in connected_edges]))

    rand_num = random()
    cumulative_prob = 0
    for i, edge in enumerate(connected_edges):
        cumulative_prob += probabilities[i]
        if rand_num <= cumulative_prob:
            if edge.a_vertex != ant.current_vertex:
                return edge, edge.a_vertex
            else:
                return edge, edge.b_vertex


def all_has_completed_tour(ants, target_vertex):
    for ant in ants:
        if ant.current_vertex:
            last_edge = ant.edges_travelled[-1]
            if not last_edge.has_vertex(target_vertex):
                return False
    return True


def is_shorter_path(edges_travelled, shorter_path):
    if len(edges_travelled) < len(shorter_path):
        shorter_path = edges_travelled
    return shorter_path


def shortest_path(matrix, start_vertex, target_vertex, live_plot=True):
    results = []
    global_shortest_path = list(repeat(0, 9999))

    if live_plot:
        live_plot = LivePheromonePlot(matrix, start_vertex, target_vertex)

    for i in range(ant_count):
        ant = Ant(start_vertex)

        ant_at_target = False
        while not ant_at_target:
            if ant.current_vertex != target_vertex:
                edge, ant.current_vertex = next_edge_and_vertex(matrix, ant)
                ant.edges_travelled.append(edge)
            else:
                ant_at_target = True
        shorter_path = is_shorter_path(ant.edges_travelled, global_shortest_path)
        global_shortest_path = shorter_path
        results.append(ant.edges_travelled)

        progress_string = "\rProgress: {}/{}".format(i, ant_count)
        sys.stdout.write(progress_string)
        sys.stdout.flush()
        if live_plot:
            live_plot.update(matrix.edges)

    if live_plot:
        live_plot.close()

    print("\nShortest path length: {}".format(len(global_shortest_path)))
    return results, global_shortest_path

if __name__ == "__main__":
    all_path_lengths = np.zeros((iteration_count, ant_count))

    for i in range(iteration_count):
        mtrx = AcocMatrix(20, 20)
        ant_paths, _shortest_path = shortest_path(mtrx, (1, 1), (15, 15), False)
        path_lengths = [len(p) for p in ant_paths]
        all_path_lengths[i, :] = path_lengths

    plotter.draw_random(all_path_lengths.mean(0), _shortest_path, mtrx)

