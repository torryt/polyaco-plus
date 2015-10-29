import random
from collections import namedtuple
import numpy as np
import math

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


def main():
    import acoc_plotter
    from matplotlib import pyplot as plt

    r = MinMax(math.pi, 2*math.pi)
    red = semi_circle(1.0, r, 500, 0)
    acoc_plotter.plot_data(red)

    r = MinMax(0, math.pi)
    blue = semi_circle(1.0, r, 500, 1, center=(1, -.5))
    acoc_plotter.plot_data(blue)

    plt.show()


if __name__ == "__main__":
    main()
