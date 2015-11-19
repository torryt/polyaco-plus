import math
import random
from collections import namedtuple
import numpy as np
import utils

MinMax = namedtuple('MinMax', ['min', 'max'])


def uniform_rectangle(x_boundary, y_boundary, num_elements, value=0):
    points = []
    x_b = MinMax(*x_boundary)
    y_b = MinMax(*y_boundary)
    for e in range(num_elements):
        x = random.random()
        x = (x * (x_b.max - x_b.min)) + x_b.min

        y = random.random()
        y = (y * (y_b.max - y_b.min)) + y_b.min
        points.append([x, y, value])
    return np.array(points).T


def uniform_circle(radius, num_elements, value):
    points = []

    for e in range(num_elements):
        rad_diff = radius * 0.1
        r = (random.random()*2*rad_diff) - rad_diff
        o = random.random()*(2*math.pi)

        x = (radius + r) * math.cos(o)
        y = (radius + r) * math.sin(o)
        points.append([x, y, value])
    return np.array(points).T


def semi_circle(radius, circle_range, num_elements, value, center=(0, 0)):
    points = []

    for e in range(num_elements):
        rad_diff = radius * 0.1
        r = (random.random()*2*rad_diff) - rad_diff
        o = random.random() * (circle_range.max - circle_range.min) + circle_range.min

        x = (radius + r) * math.cos(o)
        y = (radius + r) * math.sin(o)
        points.append([x, y, value])

    array = np.array(points).T
    array[0] = array[0] + center[0]
    array[1] = array[1] + center[1]
    return array


def generate_data_sets(size=500):
    sets = {}
    # sets = np.zeros([3, 3, size*2])
    r = MinMax(math.pi, 2*math.pi)
    red = semi_circle(1.0, r, size, 0)
    r = MinMax(0, math.pi)
    blue = semi_circle(1.0, r, size, 1, center=(1, -.5))
    sets['semicircle'] = np.concatenate((red, blue), axis=1)

    red = uniform_circle(1.0, size, 0)
    blue = uniform_circle(2.0, size, 1)
    sets['circle'] = np.concatenate((red, blue), axis=1)

    red = uniform_rectangle((1, 3), (2, 4), size, 0)
    blue = uniform_rectangle((4, 6), (2, 4), size, 1)
    sets['rectangle'] = np.concatenate((red, blue), axis=1)
    return sets


def load_data():
    import pickle
    return pickle.load(open('data_sets.pickle', 'rb'), encoding='latin1')


def main():
    sets = generate_data_sets()
    utils.save_object(sets, 'data_sets')


if __name__ == "__main__":
    main()
