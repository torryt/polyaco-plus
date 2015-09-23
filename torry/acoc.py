from acoc_matrix import AcocMatrix
from random import random
from itertools import repeat

ant_count = 10
iterations = 2
pheromone_constant = 1.0
decay_constant = 0.2
ant_paths = []


def choose_next_vertex(matrix, vertex):
    connected_edges = matrix.get_connected_edges(vertex)
    p_normalize = 1.0 / sum([edge.pheromone_strength for edge in connected_edges])
    for edge in connected_edges:
        edge.prob = edge.pheromone_strength * p_normalize

    p = random()
    cumulative_prob = 0
    for edge in connected_edges:
        cumulative_prob += edge.prob
        if p <= cumulative_prob:
            return edge.a_vertex if edge.a_vertex != vertex else edge.b_vertex


def all_has_completed_tour(paths, target_vertex):
    for path in paths:
        if path[-1] != target_vertex:
            return False
    return True


def put_pheromones(matrix, paths, target_vertex):

    for path in paths:

        path_length = len(path)
        current_vertex_index = 1
        last_vertex = False
        while not last_vertex:

            for edge in matrix.edges:
                if edge.have_vertices(path[current_vertex_index], path[current_vertex_index + 1]):
                    edge.pheromone_strength = pheromone_constant / path_length
                    break
            current_vertex_index += 1
            if path[current_vertex_index] == target_vertex:
                last_vertex = True


def pheromones_decay(matrix):
    for edge in matrix:
        edge.pheromone_strength *= (1-pheromone_constant)


def shortest_path(matrix, start_vertex, target_vertex):
    for it in range(iterations):
        paths = [[start_vertex]] * ant_count
        while not all_has_completed_tour(paths, target_vertex):
            for ant in range(ant_count):
                if not paths[ant][-1] == target_vertex:
                    paths[ant].append(choose_next_vertex(matrix, paths[ant]))

        put_pheromones(matrix, paths, target_vertex)
        pheromones_decay(matrix)


if __name__ == "__main__":
    shortest_path(AcocMatrix(10, 10), (1, 1), (8, 8))
