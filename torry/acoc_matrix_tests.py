import unittest
from acoc_matrix import AcocEdge, AcocMatrix


class TestEdge(unittest.TestCase):

    def test_have_vertices_returns_true_on_identical_vertices(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)
        edge = AcocEdge(vertex_a, vertex_b)
        self.assertTrue(edge.has_vertex(vertex_a, vertex_b))

    def test_have_vertices_returns_true_on_identical_vertices_but_reversed(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)
        edge = AcocEdge(vertex_a, vertex_b)
        self.assertTrue(edge.has_vertex(vertex_b, vertex_a))

    def test_have_vertices_returns_false_if_not_both_vertices_are_there(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)
        edge = AcocEdge(vertex_a, vertex_b)
        self.assertFalse(edge.has_vertex(vertex_a, vertex_a))

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


class TestGetConnectedEdges(unittest.TestCase):
    def test_connected_edges_returns_two_when_vertex_is_corner_vertex(self):
        matrix = AcocMatrix(4, 4)
        connected_edges_count = len(matrix.get_connected_edges((0, 0)))
        self.assertEqual(2, connected_edges_count)

    def test_connected_edges_returns_three_when_vertex_is_perimeter_vertex(self):
        matrix = AcocMatrix(4, 4)
        connected_edges_count = len(matrix.get_connected_edges((0, 1)))
        self.assertEqual(3, connected_edges_count)

    def test_connected_edges_returns_four_when_vertex_is_middle_vertex(self):
        matrix = AcocMatrix(4, 4)
        connected_edges_count = len(matrix.get_connected_edges((1, 1)))
        self.assertEqual(4, connected_edges_count)

if __name__ == '__main__':
    unittest.main()
