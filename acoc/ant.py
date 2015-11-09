from copy import copy

import numpy as np
from numpy.random.mtrand import choice

from utils.utils import normalize


class Ant:
    def __init__(self, start_vertex):
        self.start_vertex = start_vertex
        self.current_edge = None
        self.edges_travelled = []

    def move_ant(self):
        edges_travelled = self.edges_travelled if len(self.edges_travelled) > 0 else None
        if self.current_edge is None:
            connected_edges = self.start_vertex.connected_edges
        else:
            connected_edges = copy(self.current_edge.target.connected_edges)
            connected_edges.remove(self.current_edge.twin)
        if edges_travelled is not None:
            for e in edges_travelled:
                if e in connected_edges:
                    connected_edges.remove(e)
        if len(connected_edges) == 0:
            return None
        weights = normalize(np.array([e.pheromone_strength for e in connected_edges]))
        edge = choice(connected_edges, p=weights)
        self.current_edge = edge
        return edge
