import unittest
import numpy as np
from numba import cuda
import math

from acoc import ray_cast as rc
from acoc.ray_cast import Pt
from acoc.acoc_matrix import AcocEdge


class TestPointInside(unittest.TestCase):

    def test_ray_intersect_segment_returns_true(self):
        p = Pt(0, 1)
        e = AcocEdge(Pt(1, 1), Pt(1, 3))
        self.assertTrue(rc.ray_intersect_segment(p, e))

    def test_ray_intersect_segment_returns_false(self):
        p = Pt(2, 2)
        e = AcocEdge(Pt(1, 1), Pt(1, 3))
        self.assertFalse(rc.ray_intersect_segment(p, e))


class TestRayIntersectSegmentCuda(unittest.TestCase):

    def get_data(self):
        P = np.array([[1, 1]], dtype='float32')
        e1 = [[1, 0], [1, 3]]
        e2 = [[1, 3], [-2, 0]]
        E = np.array([e1, e2], dtype='float32')
        return P, E

    def test_function_returns_bool(self):
        P, E = self.get_data()
        threads_per_block = E.size
        blocks_per_grid = 1
        result = np.empty((threads_per_block, blocks_per_grid), dtype=bool)
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](P, E, result)
        self.assertTrue(result[0].dtype is 'bool', 'Type is not bool but "{}" '.format(result[0].dtype))

    def test_result_array_equal_length_of_edge_array(self):
        P, E = self.get_data()
        threads_per_block = E.size
        blocks_per_grid = 1
        result = np.empty((threads_per_block, blocks_per_grid), dtype=bool)
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](P, E, result)
        self.assertEqual(len(result), 2)

    def test_result_array_is_e_times_p_dim(self):
        P, E = self.get_data()
        threads_per_block = E.shape[0]
        blocks_per_grid = P.shape[0]
        result = np.empty((threads_per_block, blocks_per_grid), dtype=bool)
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](P, E, result)
        self.assertEqual(result.shape, (threads_per_block, blocks_per_grid))


class TestIsPointInsideCuda(unittest.TestCase):

    def test_function_returns_array(self):
        points = np.array([[1, 1], [2, 2]], dtype='float32')
        e1 = [[1, 0], [1, 3]]
        e2 = [[1, 3], [-2, 0]]
        edges = np.array([e1, e2], dtype='float32')
        threads_per_block = points.shape[0]
        blocks_per_grid = 1
        result = np.empty(1, dtype='uint8')
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](points, edges, result)
        self.assertTrue(type(result[0]) is np.uint8, 'Type is not {} but "{}" '.format(np.float32, type(result[0])))


class TestCudaGrid(unittest.TestCase):

    def test_grid_fills_all_values(self):

        @cuda.jit
        def make_true(arr):
            x, y = cuda.grid(2)
            if x < arr.shape[0] and y < arr.shape[1]:
                arr[x, y] = True

        arr = np.empty((64, 64), dtype=bool)
        threads_per_block = (16, 16)
        block_x = math.ceil(arr.shape[0] / threads_per_block[0])
        block_y = math.ceil(arr.shape[1] / threads_per_block[1])
        blocks_per_grid = (block_x, block_y)

        make_true[blocks_per_grid, threads_per_block](arr)

        # Assert whether all values evaluate to true
        self.assertTrue(np.all(arr), 'Not all values in array is True: \n{}'.format(arr))
