import unittest
import numpy as np
from acoc import ray_cast


class TestCostFuncJit(unittest.TestCase):

    def test_ray_intersect_segment_jit_returns_bool(self):
        p = [0, 0]
        e = [[1, -1], [1, 1]]
        result = ray_cast.ray_intersect_segment_jit(p, e)
        self.assertEqual(type(result), bool)

    def test_is_point_inside_jit_returns_bool(self):
        p = [0, 0]
        E = np.array([[
            [1, -1], [1, 1]
        ]])
        result = ray_cast.is_point_inside_jit(p, E)
        self.assertEqual(type(result), bool)
