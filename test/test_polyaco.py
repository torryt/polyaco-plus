import unittest
import numpy as np

import acoc
from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.polygon import polygon_length


class TestEvaluate(unittest.TestCase):
    def setUp(self):

        self.data = np.array([[0.1, 0.1, 0.1],
                              [1, 1, 2],
                              [2, 2, 3],
                              [3, 3, 4],
                              [4, 4, 5]])
        self.target = np.array([0, 0, 1, 1, 1])

        vs = [Vertex(0, 0), Vertex(1, 0), Vertex(1, 1), Vertex(0, 1)]
        self.polygon = [
            Edge(vs[0], vs[1]),
            Edge(vs[1], vs[2]),
            Edge(vs[3], vs[2]),
            Edge(vs[0], vs[3])
        ]

    def test_evaluate_before_training_raise_value_error(self):
        clf = acoc.PolyACO(2)
        with self.assertRaises(RuntimeError):
            clf.evaluate(self.data)

    def test_evaluate_returns_list_with_equal_length_as_input(self):
        number_of_items, dimensions = self.data.shape
        clf = acoc.PolyACO(dimensions)
        clf.polygons = [self.polygon] * len(clf.planes)
        result = clf.evaluate(self.data)
        self.assertEqual(len(result), number_of_items)

    def test_evaluate_returns_1_point_of_class_1_rest_class_0(self):
        clf = acoc.PolyACO(self.data.shape[1])
        clf.polygons = [self.polygon] * len(clf.planes)
        result = clf.evaluate(self.data)
        self.assertEqual(sum(result), 4)
        class_0_count = result.count(0)
        self.assertEqual(class_0_count, 1)


class TestPolygon(unittest.TestCase):
    def test_polygon_length_four_edges_returns_4(self):
        e1 = Edge(Vertex(0, 0), Vertex(1, 0))
        e2 = Edge(Vertex(1, 0), Vertex(1, 1))
        e3 = Edge(Vertex(0, 1), Vertex(1, 1))
        e4 = Edge(Vertex(0, 0), Vertex(0, 1))

        polygon = [e1, e2, e3, e4]
        self.assertEqual(polygon_length(polygon), 4)


