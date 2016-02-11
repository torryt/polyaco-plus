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


def ray_intersect_segment(p, e):

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

ray_intersect_segment_device = cuda.jit(ray_intersect_segment, device=True)
ray_intersect_segment_vectorized = np.vectorize(ray_intersect_segment, excluded='p')


@cuda.jit
def ray_intersect_segment_cuda(P, E, result):
    point_index = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    edge_index = cuda.blockIdx.y

    p = P[point_index]
    e = E[edge_index]

    if edge_index < E.shape[0] and point_index < P.shape[0]:
        result[point_index][edge_index] = ray_intersect_segment_device(p, e)


def is_point_inside(point, solution):
    return odd(sum(ray_intersect_segment(point, edge)
                   for edge in solution))
