import unittest
import numpy as np
from datetime import datetime

import acoc
from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.polygon import polygon_length


def load_simple_polygon(start_vertex, edge_length=1):
    vs = [
        start_vertex,
        Vertex(start_vertex.x + edge_length, start_vertex.y),
        Vertex(start_vertex.x, start_vertex.y + edge_length),
        Vertex(start_vertex.x + edge_length, start_vertex.y + edge_length)
    ]
    return [Edge(vs[0], vs[1]),
            Edge(vs[0], vs[2]),
            Edge(vs[1], vs[3]),
            Edge(vs[2], vs[3])]


class TestPolyAcoClass(unittest.TestCase):
    def test_3_classes_has_3_spaces_in_each_plane_in_polygon_list(self):
        clf = acoc.PolyACO(3, [0, 1, 2])
        for plane in clf.model:
            self.assertEqual(len(plane), 3)

    def test_class_indices_with_duplicate_indices_raise_warning(self):
        with self.assertWarns(UserWarning):
            acoc.PolyACO(2, [0, 0])


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
        clf.model = [[self.polygon1, self.polygon2]] * len(clf.planes)
        result = clf.evaluate(self.data)
        self.assertEqual(len(result), self.data.shape[0])

    def test_evaluate_returns_1_point_of_class_0_rest_class_1(self):
        clf = acoc.PolyACO(self.data.shape[1], self.class_indices)
        clf.model = [[self.polygon1, self.polygon2]] * len(clf.planes)
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
        clf.model = [[self.polygon1, self.polygon2, polygon3]]
        result = list(clf.evaluate(data))
        self.assertEqual(result.count(0), 1)
        self.assertEqual(result.count(1), 2)
        self.assertEqual(result.count(2), 1)

    def test_evaluate_3_classes_3_dimensions(self):
        data = np.array([
            [0.1, 0.5, 0.1],
            [2.1, 1.1, 1.3],
            [1.1, 1.3, 1.7],
            [2.1, 2.4, 2.5]
        ])
        clf = acoc.PolyACO(data.shape[1], [0, 1, 2])
        clf.model = [[self.polygon1, self.polygon2, load_simple_polygon(Vertex(2, 2))]] * len(clf.planes)
        result = list(clf.evaluate(data))
        self.assertEqual(result.count(0), 1)
        self.assertEqual(result.count(1), 2)
        self.assertEqual(result.count(2), 1)

    def test_evaluate_3_classes_3_dimensions_with_overlap(self):
        data = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 1.5, 1.5],
            [2.5, 2.5, 2.5]
        ])
        clf = acoc.PolyACO(data.shape[1], [0, 1, 2])
        clf.model = [[self.polygon1, self.polygon2, load_simple_polygon(Vertex(2, 2))]] * len(clf.planes)
        result = list(clf.evaluate(data))
        self.assertEqual(result.count(0), 1)
        self.assertEqual(result.count(1), 1)
        self.assertEqual(result.count(2), 1)


class TestPolygon(unittest.TestCase):
    def test_polygon_length_four_edges_returns_4(self):
        e1 = Edge(Vertex(0, 0), Vertex(1, 0))
        e2 = Edge(Vertex(1, 0), Vertex(1, 1))
        e3 = Edge(Vertex(0, 1), Vertex(1, 1))
        e4 = Edge(Vertex(0, 0), Vertex(0, 1))

        polygon = [e1, e2, e3, e4]
        self.assertEqual(polygon_length(polygon), 4)
