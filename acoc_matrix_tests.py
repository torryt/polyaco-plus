import unittest
from acoc_matrix import AcocEdge


class TestEdge(unittest.TestCase):

    def test_have_vertices_returns_true_on_identical_vertices(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)

        edge = AcocEdge(vertex_a, vertex_b)
        self.assertTrue(edge.have_vertices(vertex_a, vertex_b))

    def test_have_vertices_returns_true_on_identical_vertices_but_reversed(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)

        edge = AcocEdge(vertex_a, vertex_b)
        self.assertTrue(edge.have_vertices(vertex_b, vertex_a))

    def test_have_vertices_returns_false_if_not_both_vertices_are_there(self):
        vertex_a = (1, 1)
        vertex_b = (2, 2)

        edge = AcocEdge(vertex_a, vertex_b)
        self.assertFalse(edge.have_vertices(vertex_a, vertex_a))


if __name__ == '__main__':
    unittest.main()
