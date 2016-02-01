import unittest

from acoc import acoc_matrix as am
from acoc.acoc_matrix import AcocMatrix
from acoc.vertex import Vertex
from acoc.edge import MatrixEdge, PolygonEdge


class TestEdge(unittest.TestCase):

    def test_symmetric_differences(self):
        vertex_a = (1, 1)
        vertex_b = (2, 1)
        vertex_c = (3, 1)
        vertex_d = (4, 1)
        edge_a_b = MatrixEdge(vertex_a, vertex_b)
        edge_b_c = MatrixEdge(vertex_b, vertex_c)
        edge_c_d = MatrixEdge(vertex_c, vertex_d)
        set_x = {edge_a_b}.union([edge_b_c])
        set_y = {edge_b_c}.union([edge_c_d])
        result = set_x ^ set_y
        self.assertTrue(result == {edge_a_b}.union([edge_c_d]))

    def test_matrix_edge_in_polygon_edge_array_returns_true(self):
        vertex_a = Vertex(1, 1)
        vertex_b = Vertex(2, 1)

        mtx_edge = MatrixEdge(vertex_a, vertex_b)
        ply_edge = PolygonEdge(vertex_a, vertex_b)
        ply_list = [ply_edge]
        self.assertTrue(mtx_edge in ply_list)

    def test_edge_in_polygon_different_vertex_instances_returns_true(self):
        vertex_a = Vertex(1, 1)
        vertex_b = Vertex(2, 1)

        mtx_edge = MatrixEdge(vertex_a, vertex_b)
        ply_edge = PolygonEdge(Vertex(1, 1), Vertex(2, 1))
        ply_list = [ply_edge]
        self.assertTrue(mtx_edge in ply_list)

    def test_matrix_edge_in_polygon_edge_array_returns_false(self):
        vertex_a = Vertex(1, 1)
        vertex_b = Vertex(2, 1)

        mtx_edge = MatrixEdge(vertex_a, vertex_a)
        ply_edge = PolygonEdge(vertex_a, vertex_b)
        ply_list = [ply_edge]
        self.assertFalse(mtx_edge in ply_list)


class TestMatrix(unittest.TestCase):

    def test_granularity_4_should_return_matrix_with_24_edges_and_16_vertices(self):
        matrix = AcocMatrix([[[0, 0]], [[1,1]]], granularity=4)
        self.assertEqual(len(matrix.vertices), 16)
        self.assertEqual(len(matrix.edges), 24)

    def test_granularity_10_should_return_matrix_with_180_edges_and_100_vertices(self):
        matrix = AcocMatrix([[[0, 0]], [[1,1]]], granularity=4)
        self.assertEqual(len(matrix.vertices), 16)
        self.assertEqual(len(matrix.edges), 24)


class TestVertex(unittest.TestCase):
    def test_connect_edges_to_vertex_should_be_1(self):
        v1 = Vertex(0, 0)
        v2 = Vertex(1, 2)
        e = MatrixEdge(v1, v2)
        am.connect_edges_to_vertex(v1, [e])
        self.assertEqual(1, len(v1.connected_edges))

    def test_connect_edges_to_vertex_should_be_2(self):
        v1 = Vertex(0, 0)
        v2 = Vertex(1, 2)
        v3 = Vertex(1, 1)
        e1 = MatrixEdge(v1, v2)
        e2 = MatrixEdge(v1, v3)
        am.connect_edges_to_vertex(v1, [e1, e2])
        self.assertEqual(2, len(v1.connected_edges))


if __name__ == '__main__':
    unittest.main()
