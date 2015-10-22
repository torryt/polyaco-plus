import unittest
from torry.acoc_matrix import AcocEdge, AcocMatrix, Vertex
import torry.acoc_matrix as am


class TestEdge(unittest.TestCase):

    def test_have_vertices_returns_true_on_identical_vertices(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)
        edge = AcocEdge(vertex_a, vertex_b)
        self.assertTrue(edge.has_both_vertices(vertex_a, vertex_b))

    def test_have_vertices_returns_true_on_identical_vertices_but_reversed(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)
        edge = AcocEdge(vertex_a, vertex_b)
        self.assertTrue(edge.has_both_vertices(vertex_b, vertex_a))

    def test_have_vertices_returns_false_if_not_both_vertices_are_there(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)
        edge = AcocEdge(vertex_a, vertex_b)
        self.assertFalse(edge.has_both_vertices(vertex_a, vertex_a))

    def test_edge_count_should_be_12_in_3_by_3_matrix(self):
        matrix = AcocMatrix(3, 3)
        self.assertEqual(12, len(matrix.edges))

    def test_edge_count_should_be_4_in_2_by_2_matrix(self):
        matrix = AcocMatrix(2, 2)
        self.assertEqual(4, len(matrix.edges))

    def test_edge_count_should_be_7_in_2_by_3_matrix(self):
        matrix = AcocMatrix(2, 3)
        self.assertEqual(7, len(matrix.edges))

    def test_vertex_count_should_be_4_in_2_by_2_matrix(self):
        vertices = AcocMatrix(2, 2).vertices
        self.assertEqual(4, len(vertices))

    def test_vertex_count_should_be_6_in_2_by_3_matrix(self):
        vertices = AcocMatrix(2, 3).vertices
        self.assertEqual(6, len(vertices))


class TestVertex(unittest.TestCase):
    def test_connect_edges_to_vertex_should_be_1(self):
        v1 = Vertex(0, 0)
        v2 = Vertex(1, 2)
        e = AcocEdge(v1, v2)
        am.connect_edges_to_vertex(v1, [e])
        self.assertEqual(1, len(v1.connected_edges))

    def test_connect_edges_to_vertex_should_be_2(self):
            v1 = Vertex(0, 0)
            v2 = Vertex(1, 2)
            v3 = Vertex(1, 1)
            e1 = AcocEdge(v1, v2)
            e2 = AcocEdge(v1, v3)
            am.connect_edges_to_vertex(v1, [e1, e2])
            self.assertEqual(2, len(v1.connected_edges))


if __name__ == '__main__':
    unittest.main()
