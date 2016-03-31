import numpy as np


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
