import unittest

from acoc.edge import Edge
from acoc.vertex import Vertex
from acoc.polygon import polygon_length


class TestPolygon(unittest.TestCase):
    def test_polygon_length_four_edges_returns_4(self):
        e1 = Edge(Vertex(0, 0), Vertex(1, 0))
        e2 = Edge(Vertex(1, 0), Vertex(1, 1))
        e3 = Edge(Vertex(0, 1), Vertex(1, 1))
        e4 = Edge(Vertex(0, 0), Vertex(0, 1))

        polygon = [e1, e2, e3, e4]
        self.assertEqual(polygon_length(polygon), 4)
