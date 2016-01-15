from collections import namedtuple
import sys
from itertools import groupby
from numba import vectorize
import numpy as np

_eps = 0.00001
_huge = sys.float_info.max
_tiny = sys.float_info.min
Pt = namedtuple('Point', ['x', 'y'])


def _odd(x):
    return x % 2 == 1


def ray_intersect_segment(p, e):

    a = e.start
    b = e.target

    if a.y > b.y:
        a, b = b, a

    if p.item(1) == a.y or p.item(1) == b.y:
        p = Pt(p.item(0), p.item(1) + _eps)

    if p.item(1) > b.y or p.item(1) < a.y:
        return False
    if p.item(0) > max(a.x, b.x):
        return False

    if p.item(0) < min(a.x, b.x):
        return True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.item(0)) > _tiny:
            m_blue = (p.item(1) - a.y) / float(p.item(0) - a.x)
        else:
            m_blue = _huge
        return m_blue >= m_red


def is_point_inside(vertex, solution):
    np_vertex = np.array([vertex[0], vertex[1]])
    np_polygon = np.array(solution)

    # p_copy = Pt(vertex[0], vertex[1])
    return _odd(sum(ray_intersect_segment(np_vertex, edge)
                    for edge in np_polygon))
