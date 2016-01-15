from collections import namedtuple
import sys
from numba import cuda
import numpy as np

_eps = 0.00001
_huge = sys.float_info.max
_tiny = sys.float_info.min
Pt = namedtuple('Point', ['x', 'y'])


def odd(x):
    return x % 2 == 1


def _ray_intersect_segment(p, e):
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


# @cuda.jit
# def ray_intersect_segment_cuda(p, E, result):
#     tx = cuda.threadIdx.x
#
#     EPS = 0.00001
#     HUGE = 9000000000000000
#     TINY = 0.000000000000001
#
#     e = E[tx]
#     a = e[0]
#     b = e[1]
#
#     if a[1] > b[1]:
#         a, b = b, a
#
#     if p[1] == a[1] or p[1] == b[1]:
#         p[1] += EPS
#
#     if p[1] > b[1] or p[1] < a[1]:
#         result[tx] = False
#         return
#     if p[0] > max(a[0], b[0]):
#         result[tx] = False
#         return
#
#     if p[0] < min(a[0], b[0]):
#         result[tx] = True
#         return
#     else:
#         if abs(a[0] - b[0]) > TINY:
#             m_red = (b[1] - a[1]) / float(b[0] - a[0])
#         else:
#             m_red = HUGE
#         if abs(a[0] - p[0]) > TINY:
#             m_blue = (p[1] - a[1]) / float(p[0] - a[0])
#         else:
#             m_blue = HUGE
#         result[tx] = m_blue >= m_red
#         return


@cuda.jit(device=True)
def ray_intersect_segment_device(p, e):

    EPS = 0.00001
    HUGE = 9000000000000000
    TINY = 0.000000000000001

    a = e[0]
    b = e[1]

    if a[1] > b[1]:
        a, b = b, a

    if p[1] == a[1] or p[1] == b[1]:
        p[1] += EPS

    if p[1] > b[1] or p[1] < a[1]:
        return False
    if p[0] > max(a[0], b[0]):
        return False
    if p[0] < min(a[0], b[0]):
        return True
    else:
        if abs(a[0] - b[0]) > TINY:
            m_red = (b[1] - a[1]) / float(b[0] - a[0])
        else:
            m_red = HUGE
        if abs(a[0] - p[0]) > TINY:
            m_blue = (p[1] - a[1]) / float(p[0] - a[0])
        else:
            m_blue = HUGE
        return m_blue >= m_red


@cuda.jit
def ray_intersect_segment_cuda(E, P, result):
    edge_index, point_index = cuda.grid(2)
    e = E[edge_index]
    p = P[point_index]

    if edge_index < E.shape[0] and point_index < P.shape[0]:
        result[edge_index][point_index] = ray_intersect_segment_device(p, e)


def is_point_inside_cuda(points, solution):
    pass


def is_point_inside(vertex, solution):
    p_copy = Pt(vertex[0], vertex[1])
    return odd(sum(_ray_intersect_segment(p_copy, edge)
                   for edge in solution))
