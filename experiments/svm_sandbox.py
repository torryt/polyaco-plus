from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

from utils import data_generator as dg

MinMax = namedtuple('MinMax', ['min', 'max'])


def run():
    # red = dg.semi_circle(1.0, MinMax(math.pi, 2*math.pi), 500, 0)
    # blue = dg.semi_circle(1.0, MinMax(0, math.pi), 500, 1, center=(1, -.5))

    # red = dg.uniform_rectangle((1, 3), (2, 4), 500, 0)
    # blue = dg.uniform_rectangle((4, 6), (2, 4), 500, 1)

    red = dg.uniform_circle(3.0, 500, 0)
    blue = dg.uniform_circle(2.0, 500, 1)
    # green = dg.uniform_circle(1.0, 500, 2)

    data = np.concatenate((red, blue), axis=1)
    # data = np.concatenate((data, green), axis=1)
    X = data[:2].T
    y = data[2]

    h = .02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    plt.subplot(111)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig('svm_sandbox.eps')

    # plt.show()

run()
