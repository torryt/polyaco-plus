from collections import namedtuple
import sys
from itertools import groupby

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

    if p.y == a.y or p.y == b.y:
        p = Pt(p.x, p.y + _eps)

    if p.y > b.y or p.y < a.y:
        return False
    if p.x > max(a.x, b.x):
        return False

    if p.x < min(a.x, b.x):
        return True
    else:
        if abs(a.x - b.x) > _tiny:
            m_red = (b.y - a.y) / float(b.x - a.x)
        else:
            m_red = _huge
        if abs(a.x - p.x) > _tiny:
            m_blue = (p.y - a.y) / float(p.x - a.x)
        else:
            m_blue = _huge
        return m_blue >= m_red


def is_point_inside(vertex, solution):
    p_copy = Pt(vertex[0], vertex[1])
    return _odd(sum(ray_intersect_segment(p_copy, edge)
                    for edge in solution))