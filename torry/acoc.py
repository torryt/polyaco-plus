from acoc_matrix import AcocMatrix, AcocEdge
from random import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat


ant_count = 10
iterations = 100
pheromone_constant = 1.0
decay_constant = 0.1


def normalize_0_to_1(values):
    if values.sum() == 0.0:
        return [1.0 / len(values)] * len(values)
    normalize_const = 1.0 / values.sum()
    return values * normalize_const


def choose_next_vertex(matrix, vertex):
    connected_edges = matrix.get_connected_edges(vertex)
    probabilities = normalize_0_to_1(np.array([e.pheromone_strength for e in connected_edges]))
    p = random()
    cumulative_prob = 0
    for i, edge in enumerate(connected_edges):
        cumulative_prob += probabilities[i]
        if p <= cumulative_prob:
            return edge.a_vertex if edge.a_vertex != vertex else edge.b_vertex


def all_has_completed_tour(paths, target_vertex):
    for path in paths:
        if path[-1] != target_vertex:
            return False
    return True


def put_pheromones(matrix, path, target_vertex):
    path_length = len(path)
    current_vertex_index = 0
    last_vertex = False
    while not last_vertex:
        for edge in matrix.edges:
            if edge.has_vertex(path[current_vertex_index], path[current_vertex_index + 1]):
                edge.pheromone_strength = pheromone_constant / path_length
                break
        current_vertex_index += 1
        if path[current_vertex_index] == target_vertex:
            last_vertex = True


def pheromones_decay(matrix):
    for edge in matrix.edges:
        edge.pheromone_strength *= (1-pheromone_constant)


def iteration_result(matrix, paths):
    return np.array([len(path) for path in paths]).mean()


def show_plot_results(results):
    x_coord = range(0, len(results))
    y_coord = results
    plt.plot(x_coord, y_coord)
    plt.axis([0, max(x_coord), 0, max(y_coord)])
    plt.show()


def has_shorter_path(paths, global_shortest_path):
    for p in paths:
        if len(p) < len(global_shortest_path):
            global_shortest_path = p
            return True, global_shortest_path
    return False, global_shortest_path


def convert_path_to_edge_list(path):
    edges = []
    for i,_ in enumerate(path):
        if i < len(path)-1:
            edges.append(AcocEdge(path[i], path[i+1]))
    return edges


def shortest_path(matrix, start_vertex, target_vertex):
    results = []
    global_shortest_path = list(repeat(0, 2000))

    for iteration in range(iterations):
        paths = [[start_vertex]] * ant_count
        while not all_has_completed_tour(paths, target_vertex):
            for ant in range(ant_count):
                if not paths[ant][-1] == target_vertex:
                    paths[ant].append(choose_next_vertex(matrix, paths[ant][-1]))

        is_shorter_path = has_shorter_path(paths, global_shortest_path)
        if is_shorter_path[0]:
            global_shortest_path = is_shorter_path[1]
            put_pheromones(matrix, global_shortest_path, target_vertex)

        pheromones_decay(matrix)
        iter_result = iteration_result(matrix, paths)
        results.append(iter_result)
        if iteration % 10 == 0:
            print("Iteration {} avg. path length: {}".format(iteration, iter_result))
    show_plot_results(results)
    print(convert_path_to_edge_list(global_shortest_path))

    return results


if __name__ == "__main__":
    shortest_path(AcocMatrix(10, 10), (1, 1), (8, 8))
