import unittest

import is_point_inside as ipi
from acoc_matrix import AcocEdge, Vertex


class TestPointInside(unittest.TestCase):

    def test_ray_intersect_segment_returns_true(self):
        p = Vertex(0, 2)
        e = AcocEdge((1, 1), (1, 3))
        self.assertTrue(ipi.ray_intersect_segment(p, e))

    def test_ray_intersect_segment_returns_false(self):
        p = Vertex(2, 2)
        e = AcocEdge((1, 1), (1, 3))
        self.assertFalse(ipi.ray_intersect_segment(p, e))