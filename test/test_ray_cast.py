import unittest
import numpy as np

from acoc import ray_cast as rc
from acoc.ray_cast import Pt
from acoc.acoc_matrix import AcocEdge


class TestPointInside(unittest.TestCase):

    def test_ray_intersect_segment_returns_true(self):
        p = Pt(0, 1)
        e = AcocEdge(Pt(1, 1), Pt(1, 3))
        self.assertTrue(rc._ray_intersect_segment(p, e))

    def test_ray_intersect_segment_returns_false(self):
        p = Pt(2, 2)
        e = AcocEdge(Pt(1, 1), Pt(1, 3))
        self.assertFalse(rc._ray_intersect_segment(p, e))


class TestRayIntersectSegmentCuda(unittest.TestCase):

    def test_function_returns_bool(self):
        p = np.array([1, 1], dtype='float32')
        e1 = [[1, 0], [1, 3]]
        e2 = [[1, 3], [-2, 0]]
        E = np.array([e1, e2], dtype='float32')
        threads_per_block = E.size
        blocks_per_grid = 1
        result = np.empty(E.shape[0], dtype='bool_')
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](p, E, result)
        self.assertTrue(type(result[0]) is np.bool_, 'Type is not bool but "{}" '.format(type(result[0])))

    def test_result_array_equal_length_of_edge_array(self):
        p = np.array([1, 1], dtype='float32')
        e1 = [[1, 0], [1, 3]]
        e2 = [[1, 3], [-2, 0]]
        E = np.array([e1, e2], dtype='float32')
        threads_per_block = E.size
        blocks_per_grid = 1
        result = np.empty(E.shape[0], dtype='bool_')
        rc.ray_intersect_segment_cuda[blocks_per_grid, threads_per_block](p, E, result)
        self.assertEqual(len(result), 2)



# threadsperblock = 32
# blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock


    # def test_ray_parallel_returns_false_on_horizontal_edges(self):
    #     p = np.array([1,1])
    #     e1 = [[1, 2], [2, 2]]
    #     e2 = [[1, 0], [1, 3]]
    #     E = np.array([0, 1, 3, 4], dtype='float32')
    #     self.assertEqual(rc.ray_intersect_segment_parallel(p, E).shape[0], 2)