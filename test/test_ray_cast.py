import unittest
import numpy as np
from numba import cuda
import math

from acoc import ray_cast as rc
from acoc.ray_cast import Pt
from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.polygon import polygon_to_array


class TestPointInside(unittest.TestCase):
    def test_ray_intersect_segment_returns_true(self):
        p = [0, 1]
        e = [[1, 1], [1, 3]]
        self.assertTrue(rc.ray_intersect_segment(p, e))

    def test_ray_intersect_segment_returns_false(self):
        p = Pt(2, 2)
        e = [[1, 1], [1, 3]]
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
        self.assertTrue(type(result[0][0]) is np.bool_, 'Type is not {} but {} '.format(np.bool_, type(result[0][0])))

    def test_result_array_is_e_times_p_dim(self):
        P, E = self.get_data()
        threads_per_block = E.shape[0]
        blocks_per_grid = P.shape[0]
        result = np.empty((threads_per_block, blocks_per_grid), dtype=bool)
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](P, E, result)
        self.assertEqual(result.shape, (threads_per_block, blocks_per_grid))


class TestIsPointInsideCuda(unittest.TestCase):
    def setUp(self):
        vs = [Vertex(0, 0), Vertex(1, 0), Vertex(1, 1), Vertex(0, 1)]
        self.polygon = polygon_to_array([
            Edge(vs[0], vs[1]),
            Edge(vs[1], vs[2]),
            Edge(vs[3], vs[2]),
            Edge(vs[0], vs[3])
        ])

    def test_function_returns_array_of_floats(self):
        points = np.array([[1, 1], [2, 2]], dtype='float32')
        e1 = [[1, 0], [1, 3]]
        e2 = [[1, 3], [-2, 0]]
        edges = np.array([e1, e2], dtype='float32')
        threads_per_block = points.shape[0]
        blocks_per_grid = 1
        result = np.empty((points.shape[0], edges.shape[0]), dtype=bool)
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](points, edges, result)
        self.assertTrue(type(result[0][0]) is np.bool_, 'Type is not {} but "{}" '.format(np.bool_, type(result[0][0])))

    def test_function_returns_array_of_size_equal_to_number_of_input_points(self):
        data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        result = rc.is_points_inside_cuda(data, self.polygon)
        self.assertEqual(result.shape[0], data.shape[0])

    def test_function_returns_1_true_rest_false(self):
        data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        result = rc.is_points_inside_cuda(data, self.polygon)
        self.assertEqual(np.sum(result), 1)
        self.assertEqual(np.sum(np.invert(result)), 3)


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


class TestPointsInside(unittest.TestCase):
    def setUp(self):
        vs = [Vertex(0, 0),
              Vertex(1, 0),
              Vertex(0, 1),
              Vertex(1, 1)]
        self.edges = polygon_to_array(
            [Edge(vs[0], vs[1]),
             Edge(vs[0], vs[2]),
             Edge(vs[1], vs[3]),
             Edge(vs[2], vs[3])])

    def test_returns_list(self):
        points = np.array([[0.5, 0.5, 1], [1.0, 2.0, 0], [.7, .7, 1]])
        result = rc.points_inside(points, self.edges)
        self.assertEqual(type(result), np.ndarray)

    def test_returns_list_with_two_elements(self):
        points = np.array([[0.5, 0.5, 1], [1.0, 2.0, 0], [.7, .7, 1]])
        result = rc.points_inside(points, self.edges)
        self.assertEqual(result.shape, (2, 3))


class TestPointsOfBothClassesInside(unittest.TestCase):
    def setUp(self):
        vs = [Vertex(0, 0),
              Vertex(1, 0),
              Vertex(0, 1),
              Vertex(1, 1)]
        self.edges = polygon_to_array(
            [Edge(vs[0], vs[1]),
             Edge(vs[0], vs[2]),
             Edge(vs[1], vs[3]),
             Edge(vs[2], vs[3])])

    def test_returns_false_if_one_point_inside(self):
        points = np.array([[0.5, 0.5, 1], [1.0, 2.0, 0]])
        result = rc.points_of_both_classes_inside(points, self.edges)
        self.assertFalse(result)

    def test_returns_false_if_no_point_inside(self):
        points = np.array([[2.0, 0.5, 1], [1.0, 2.0, 0]])
        result = rc.points_of_both_classes_inside(points, self.edges)
        self.assertFalse(result)

    def test_returns_true_if_two_points_of_different_classes_inside(self):
        points = np.array([[0.5, 0.5, 1], [0.7, 0.5, 0]])
        result = rc.points_of_both_classes_inside(points, self.edges)
        self.assertTrue(result)
