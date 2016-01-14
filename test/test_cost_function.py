import unittest
import numpy as np

from acoc import cost_function as cf


class TestCostFunctionCuda(unittest.TestCase):

    def test_function_returns_float(self):
        points = np.array([[1, 1], [2, 2]], dtype='float32')
        e1 = [[1, 0], [1, 3]]
        e2 = [[1, 3], [-2, 0]]
        edges = np.array([e1, e2], dtype='float32')
        threads_per_block = edges.size
        blocks_per_grid = 1
        result = np.empty(1, dtype='float32')
        cf.cost_function_cuda[blocks_per_grid, threads_per_block](points, edges, result)
        self.assertTrue(type(result[0]) is np.float32, 'Type is not {} but "{}" '.format(np.float32, type(result[0])))