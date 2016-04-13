from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
from utils import data_manager as dg
import acoc

MinMax = namedtuple('MinMax', ['min', 'max'])


def run():
    # Based on http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
    # red = dg.gaussian_circle(3.0, 500, 0)
    # blue = dg.gaussian_circle(2.0, 500, 1)
    i = 0
    resultArray = []
    sum = 0
    average = 0

    while i < 100:
        data_set = dg.load_breast_cancer()

        # data = np.concatenate((red, blue), axis=1)
        # X = data[:2].T
        # y = data[2]

        X = data_set.data
        y = data_set.target
        h = .02

        # splitting data into training and testing set
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.33)

        # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                      np.arange(y_min, y_max, h))

        clf = svm.SVC(C=2, gamma=0.8, kernel='linear')
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        result = acoc.compute_score(predictions, y_test)
        resultArray.append(result)
        i += 1
    print(str(resultArray) + '\n')

    for i in resultArray:
        sum = sum + i
    average = sum / len(resultArray)
    print(average)




    # plt.subplot(111)

    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    #
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    # # plt.savefig('svm_sandbox.eps')
    #
    # plt.show()

run()
