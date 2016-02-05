from copy import copy
import numpy as np
from numpy.random.mtrand import choice
from utils import normalize

from acoc.edge import Edge


class Ant:
    def __init__(self, start_vertex):
        self.start_vertex = start_vertex
        self.prev_edge = None
        self.current_vertex = start_vertex
        self.edges_travelled = []
        self.is_stuck = False
        self.at_target = False

    def move_ant(self):
        connected_edges = [ce for ce in self.current_vertex.connected_edges if ce is not None]
        if self.prev_edge:
            connected_edges.remove(self.prev_edge)
        [connected_edges.remove(e) for e in self.edges_travelled if e in connected_edges]

        if len(connected_edges) == 0 or len(self.edges_travelled) > 10000:
            self.is_stuck = True
            return
        weights = normalize(np.array([e.pheromone_strength for e in connected_edges]))
        edge = choice(connected_edges, p=weights)
        self.current_vertex = edge.a if self.current_vertex != edge.a else edge.b

        self.edges_travelled.append(Edge(edge.a, edge.b))
        self.prev_edge = edge
        if self.current_vertex == self.start_vertex:
            self.at_target = True
