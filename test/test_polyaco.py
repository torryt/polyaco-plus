import unittest
import numpy as np
from datetime import datetime

import acoc
from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.polygon import polygon_length, load_simple_polygon


class TestPolyAcoClass(unittest.TestCase):
    def test_3_dim_3_class_has_9_polygons(self):
        clf = acoc.PolyACO(3, [0, 1, 2])
        polygon_count = 0
        for p in clf.polygons:
            polygon_count += len(p)
        self.assertEqual(polygon_count, 9)

    def test_4_dim_3_class_has_18_polygons(self):
        clf = acoc.PolyACO(4, [0, 1, 2])
        polygon_count = 0
        for p in clf.polygons:
            polygon_count += len(p)
        self.assertEqual(polygon_count, 18)


class TestEvaluate(unittest.TestCase):
    def setUp(self):

        self.data = np.array([[0.1, 0.1, 0.1],
                              [1.3, 1.2, 1.2],
                              [1.1, 1.1, 1.1],
                              [1.1, 1.1, 1.1],
                              [1.5, 2, 1.2]])
        self.polygon1 = load_simple_polygon(Vertex(0, 0))
        self.polygon2 = load_simple_polygon(Vertex(1, 1))
        self.class_indices = [0, 1]

    def test_evaluate_before_training_raise_value_error(self):
        clf = acoc.PolyACO(2, self.class_indices)
        with self.assertRaises(RuntimeError):
            clf.evaluate(self.data)

    def test_evaluate_returns_list_with_equal_length_as_input(self):
        clf = acoc.PolyACO(self.data.shape[1], self.class_indices)
        clf.polygons = [[self.polygon1, self.polygon2]] * len(clf.planes)
        result = clf.evaluate(self.data)
        self.assertEqual(len(result), self.data.shape[0])

    def test_evaluate_returns_1_point_of_class_0_rest_class_1(self):
        clf = acoc.PolyACO(self.data.shape[1], self.class_indices)
        clf.polygons = [[self.polygon1, self.polygon2]] * len(clf.planes)
        result = clf.evaluate(self.data)
        self.assertEqual(sum(result), 4)
        class_0_count = list(result).count(0)
        self.assertEqual(class_0_count, 1)

    def test_evaluate_3_classes_2_dimensions(self):
        data = np.array([
            [0.1, 0.5],
            [1.1, 0.5],
            [1.1, 1.3],
            [1.1, 1.4]
        ])
        clf = acoc.PolyACO(data.shape[1], [0, 1, 2])
        polygon3 = load_simple_polygon(Vertex(1, 0))
        clf.polygons = [[self.polygon1, self.polygon2, polygon3]]
        result = list(clf.evaluate(data))
        self.assertEqual(result.count(0), 1)
        self.assertEqual(result.count(1), 2)
        self.assertEqual(result.count(2), 1)

    def test_evaluate_3_classes_3_dimensions(self):
        data = np.array([
            [0.1, 0.5, 0.1],
            [1.1, 1.1, 1.3],
            [1.1, 1.3, 1.7],
            [2.1, 2.4, 2.5]
        ])
        clf = acoc.PolyACO(data.shape[1], [0, 1, 2])
        clf.polygons = [[self.polygon1, self.polygon2, load_simple_polygon(Vertex(2, 2))]]
        result = list(clf.evaluate(data))
        self.assertEqual(result.count(0), 1)
        self.assertEqual(result.count(1), 2)
        self.assertEqual(result.count(2), 1)


class TestPolygon(unittest.TestCase):
    def test_polygon_length_four_edges_returns_4(self):
        e1 = Edge(Vertex(0, 0), Vertex(1, 0))
        e2 = Edge(Vertex(1, 0), Vertex(1, 1))
        e3 = Edge(Vertex(0, 1), Vertex(1, 1))
        e4 = Edge(Vertex(0, 0), Vertex(0, 1))

        polygon = [e1, e2, e3, e4]
        self.assertEqual(polygon_length(polygon), 4)


