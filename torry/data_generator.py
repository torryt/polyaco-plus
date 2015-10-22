import random
from collections import namedtuple
import numpy as np

MinMax = namedtuple('MinMax', ['min', 'max'])


def uniform_rectangle(x_boundary, y_boundary, num_elements):
    points = []
    x_b = MinMax(*x_boundary)
    y_b = MinMax(*y_boundary)
    for e in range(num_elements):
        x = random.random()
        x = (x * (x_b.max - x_b.min)) + x_b.min

        y = random.random()
        y = (y * (y_b.max - y_b.min)) + y_b.min
        points.append([x, y])
    return np.array(points).T
