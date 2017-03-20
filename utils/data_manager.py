import math
import random
from collections import namedtuple
import numpy as np
import pickle
from bunch import Bunch
import os.path as osp
from sklearn import datasets

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


def gaussian_circle(radius, num_elements, value, spread=0.1):
    points = []

    for e in range(num_elements):
        rad = np.random.normal(radius, spread)
        o = random.random()*(2*math.pi)
        x = (radius + rad) * math.cos(o)
        y = (radius + rad) * math.sin(o)
        points.append([x, y, value])
    return np.array(points).T


def uniform_circle(radius, num_elements, value, spread=0.1):
    points = []

    for e in range(num_elements):
        rad_diff = radius * spread
        rad = (random.random()*2*rad_diff) - rad_diff
        o = random.random()*(2*math.pi)

        x = (radius + rad) * math.cos(o)
        y = (radius + rad) * math.sin(o)
        points.append([x, y, value])
    return np.array(points).T


def semi_circle_gaussian(radius=1.0, c_range=None, size=500, center=(0, 0), spread=0.3):
    points = []
    if c_range is None:
        c_range = MinMax(math.pi, 2 * math.pi)
    for _ in range(size):
        rad = np.random.normal(radius, spread)
        o = random.random() * (c_range.max - c_range.min) + c_range.min
        x = (radius + rad) * math.cos(o)
        y = (radius + rad) * math.sin(o)
        points.append([x, y])

    array = np.array(points).T
    array[0] += center[0]
    array[1] += center[1]
    return array.T


def generate_sm_dataset(size=1000):
    el_per_class = int(size / 2)
    a = semi_circle_gaussian(size=el_per_class)
    c_range = MinMax(0, math.pi)
    b = semi_circle_gaussian(size=el_per_class, c_range=c_range, center=(1.8, -.7))
    data = np.concatenate((a, b), axis=0)
    target = np.zeros(1000, dtype=int)
    target[:499] = 0
    target[500:] = 1
    return Bunch(**{'data': data, 'target': target})


def generate_data_sets(size=500):
    sets = {}
    # sets = np.zeros([3, 3, size*2])
    r = MinMax(math.pi, 2*math.pi)
    white = semi_circle_gaussian(1.0, r, size, 0)
    r = MinMax(0, math.pi)
    blue = semi_circle_gaussian(1.0, r, size, (1, -.5))
    sets['semicircle'] = np.concatenate((white, blue), axis=1)

    white = gaussian_circle(1.0, size, 0)
    blue = gaussian_circle(2.0, size, 1)
    sets['circle'] = np.concatenate((white, blue), axis=1)

    white = uniform_rectangle((1, 3), (2, 4), size, 0)
    blue = uniform_rectangle((4, 6), (2, 4), size, 1)
    sets['rectangle'] = np.concatenate((white, blue), axis=1)
    return sets


def generate_rectangle_set(size):
    white = uniform_rectangle((1, 3), (2, 4), size, 0)
    blue = uniform_rectangle((4, 6), (2, 4), size, 1)
    return np.concatenate((white, blue), axis=1)


def generate_various_sized_rectangles(sizes):
    sets = _load_data()
    for s in sizes:
        white = uniform_rectangle((1, 3), (2, 4), s, 0)
        blue = uniform_rectangle((4, 6), (2, 4), s, 1)
        sets['r_' + str(s)] = np.concatenate((white, blue), axis=1)
    pickle.dump(sets, open('data_sets.pickle', 'wb'))


def load_breast_cancer():
    import csv
    bc = Bunch()
    fn = osp.join(osp.dirname(__file__), 'breast-cancer-wisconsin.csv')
    with open(fn, 'r') as csvfile:
        data = np.array(list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)))
        mask = np.invert(np.isnan(np.sum(data, axis=1)))
        data = data[mask]
        bc.target = data[:, -1]
        bc.data = data[:, 1:-1]
        return bc


def load_wine():
    import csv
    bc = Bunch()
    fn = osp.join(osp.dirname(__file__), 'wine.csv')
    with open(fn, 'r') as csvfile:
        data = np.array(list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)))
        mask = np.invert(np.isnan(np.sum(data, axis=1)))
        data = data[mask]
        bc.target = data[:, 0]
        bc.data = np.delete(data, 0, 1)
        return bc


def load_tae():
    bc = Bunch()
    fn = osp.join(osp.dirname(__file__), 'tae.csv')
    with open(fn, 'r') as csvfile:
        data = np.array(list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)))
        bc.target = data[:, -1]
        bc.data = data[:, 0:-1]
        return bc


def load_balance_scale():
    bc = Bunch()
    fn = osp.join(osp.dirname(__file__), 'balance-scale.csv')
    with open(fn, 'r') as csvfile:
        data = np.array(list(csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)))
        bc.target = data[:, 0]
        bc.data = data[:, 1:]
        return bc


def load_australian_credit():
    import csv
    bc = Bunch()
    fn = osp.join(osp.dirname(__file__), 'aus.dat')
    with open(fn, 'r') as csvfile:
        data = np.array(list(csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)))
        mask = np.invert(np.isnan(np.sum(data, axis=1)))
        data = data[mask]
        bc.target = data[:, -1]
        bc.data = data[:, 0:-1]
        return bc


def load_data_set(name):
    if name == 'iris':
        return datasets.load_iris()
    if name == 'breast_cancer':
        return load_breast_cancer()
    if name == 'wine':
        return load_wine()
    if name == 'tae':
        return load_tae()
    if name == 'bal':
        return load_balance_scale()
    if name == 'aus':
        return load_australian_credit()
    if name == 'digits':
        return datasets.load_digits()
    if name == 'german-credit':
        return pickle.load(open(osp.join(osp.abspath(osp.dirname(__file__)), 'german-credit.pickle'), 'rb'))
    return _load_data()['name']


def _load_data():
    return pickle.load(open('utils/data_sets.pickle', 'rb'), encoding='latin1')


def list_datasets():
    keys = pickle.load(open('utils/data_sets.pickle', 'rb'), encoding='latin1').keys()
    for key in keys:
        print(key)


def main():
    sets = generate_data_sets()
    utils.save_object(sets, 'data_sets')


if __name__ == "__main__":
    class_one = uniform_rectangle((0, 2), (3, 5), 150, 0)
    class_two = uniform_rectangle((4, 6), (0, 2), 150, 1)
    dataset = np.concatenate((class_one, class_two), axis=1)
