import unittest
import numpy as np

from acoc import acoc_matrix as am
from acoc.acoc_matrix import AcocMatrix
from acoc.acoc_matrix import DIRECTION
from acoc.vertex import Vertex
from acoc.edge import Edge
import acoc.acoc_plotter as plotter


class TestEdge(unittest.TestCase):

    def test_symmetric_differences(self):
        vertex_a = (1, 1)
        vertex_b = (2, 1)
        vertex_c = (3, 1)
        vertex_d = (4, 1)
        edge_a_b = Edge(vertex_a, vertex_b)
        edge_b_c = Edge(vertex_b, vertex_c)
        edge_c_d = Edge(vertex_c, vertex_d)
        set_x = {edge_a_b}.union([edge_b_c])
        set_y = {edge_b_c}.union([edge_c_d])
        result = set_x ^ set_y
        self.assertTrue(result == {edge_a_b}.union([edge_c_d]))

    def test_matrix_edge_in_polygon_edge_array_returns_true(self):
        vertex_a = Vertex(1, 1)
        vertex_b = Vertex(2, 1)

        mtx_edge = Edge(vertex_a, vertex_b)
        ply_edge = Edge(vertex_a, vertex_b)
        ply_list = [ply_edge]
        self.assertTrue(mtx_edge in ply_list)

    def test_edge_in_polygon_different_vertex_instances_returns_true(self):
        vertex_a = Vertex(1, 1)
        vertex_b = Vertex(2, 1)

        mtx_edge = Edge(vertex_a, vertex_b)
        ply_edge = Edge(Vertex(1, 1), Vertex(2, 1))
        ply_list = [ply_edge]
        self.assertTrue(mtx_edge in ply_list)

    def test_matrix_edge_in_polygon_edge_array_returns_false(self):
        vertex_a = Vertex(1, 1)
        vertex_b = Vertex(2, 1)

        mtx_edge = Edge(vertex_a, vertex_a)
        ply_edge = Edge(vertex_a, vertex_b)
        ply_list = [ply_edge]
        self.assertFalse(mtx_edge in ply_list)


class TestMatrix(unittest.TestCase):
    def test_granularity_4_should_return_matrix_with_24_edges_and_16_vertices(self):
        matrix = AcocMatrix([[[0, 0]], [[1,1]]], granularity=4)
        self.assertEqual(len(matrix.vertices), 16)
        self.assertEqual(len(matrix.edges), 24)

    def test_all_edges_move_in_positive_direction_from_a_to_b(self):
        matrix = AcocMatrix([[0, 0], [1,1]])
        is_positive = map(lambda e: e.b.x >= e.a.x and e.b.y >= e.a.y, matrix.edges)
        self.assertTrue(all(is_positive))

    def test_all_edges_move_in_positive_direction_from_a_to_b_after_level_up(self):
        matrix = AcocMatrix([[0, 0], [1, 1]])
        matrix.level_up()
        is_positive = map(lambda e: e.b.x >= e.a.x and e.b.y >= e.a.y, matrix.edges)
        self.assertTrue(all(is_positive))

    def test_edge_length_x_is_equal_a_horizontal_edge_in_matrix(self):
        matrix = AcocMatrix([[0, 3, 5], [0, 3, 4]], granularity=10)
        horiz_e = list(filter(lambda e: e.a.y == e.b.y, matrix.edges))[0]
        e_len = horiz_e.b.x - horiz_e.a.x
        self.assertEqual(e_len, matrix.init_edge_length_x)

    def test_edge_length_x_is_equal_a_horizontal_edge_in_matrix_after_level_up(self):
        matrix = AcocMatrix([[0, 3, 5], [0, 3, 4]], granularity=10)
        matrix.level_up()
        horiz_e = list(filter(lambda e: e.a.y == e.b.y, matrix.edges))[0]
        e_len = horiz_e.b.x - horiz_e.a.x
        self.assertEqual(e_len, matrix.init_edge_length_x / 2)

    def test_first_vertex_is_smallest_vertex(self):
        matrix = AcocMatrix([[0, 3, 5], [0, 3, 4]], granularity=10)
        first_v = matrix.vertices[0]
        smallest_x = min(matrix.vertices, key=lambda v: v.x).x
        smallest_y = min(matrix.vertices, key=lambda v: v.y).y

        self.assertEqual(first_v.x, smallest_x)
        self.assertEqual(first_v.y, smallest_y)


class TestLevelUpNested(unittest.TestCase):
    def setUp(self):
        self.show = False
        self.data = np.array([[0, 0, 3], [0, 0, 3], [0, 1, 1]])

    def test_level_up_nested_increase_number_of_edges(self):
        matrix = AcocMatrix(self.data, nest_grid=True)
        old_edge_count = len(matrix.edges)
        if self.show:
            plotter.plot_matrix_and_data(matrix, matrix.data, show=True)
        matrix.level_up_nested()
        if self.show:
            plotter.plot_matrix_and_data(matrix, matrix.data, show=True)
        new_edge_count = len(matrix.edges)
        self.assertGreater(new_edge_count, old_edge_count)

    def test_level_up_nested_increase_number_of_edges_with_8(self):
        matrix = AcocMatrix(self.data, nest_grid=True)
        old_edge_count = len(matrix.edges)
        matrix.level_up_nested()
        new_edge_count = len(matrix.edges)
        self.assertEqual(new_edge_count - old_edge_count, 8)

    def test_level_up_one_section_increases_vertex_count_with_5(self):
        matrix = AcocMatrix(self.data, nest_grid=True)
        old_vertex_count = len(matrix.vertices)
        matrix.level_up_nested()
        new_vertex_count = len(matrix.vertices)
        self.assertEqual(new_vertex_count - old_vertex_count, 5)

    def test_level_up_two_times_increases_vertex_count_with_10(self):
        matrix = AcocMatrix(self.data, nest_grid=True)
        old_vertex_count = len(matrix.vertices)
        matrix.level_up_nested()
        matrix.level_up_nested()
        new_vertex_count = len(matrix.vertices)
        self.assertEqual(new_vertex_count - old_vertex_count, 10)

    def test_level_up_5_increase_vertex_count_with_5_times_5(self):
        matrix = AcocMatrix(self.data, nest_grid=True)
        old_vertex_count = len(matrix.vertices)
        for _ in range(5):
            matrix.level_up_nested()
        new_vertex_count = len(matrix.vertices)
        self.assertEqual(new_vertex_count - old_vertex_count, 5*5)

    def test_level_up_nested_increase_number_of_vertices_by_10_in_other_data_set(self):
        self.data = np.array([[0, 0, 3, 3], [0, 0, 3, 3], [0, 1, 0, 1]])
        matrix = AcocMatrix(self.data, nest_grid=True)
        old_vertex_count = len(matrix.vertices)
        if self.show:
            plotter.plot_matrix_and_data(matrix, matrix.data, show=True)
        matrix.level_up_nested()
        if self.show:
            plotter.plot_matrix_and_data(matrix, matrix.data, show=True)
        new_vertex_count = len(matrix.vertices)
        self.assertEqual(new_vertex_count - old_vertex_count, 10)

    def test_level_up_nested_increase_number_of_vertices_by_9_in_adjacent_sections_set(self):
        self.data = np.array([[0, 0, 0, 0, 3], [0, 0, 3, 3, 3], [0, 1, 0, 1, 1]])
        matrix = AcocMatrix(self.data, nest_grid=True)
        old_vertex_count = len(matrix.vertices)
        if self.show:
            plotter.plot_matrix_and_data(matrix, matrix.data, show=True)
        matrix.level_up_nested()
        if self.show:
            plotter.plot_matrix_and_data(matrix, matrix.data, show=True)
        new_vertex_count = len(matrix.vertices)
        self.assertEqual(new_vertex_count - old_vertex_count, 9)


class TestVertex(unittest.TestCase):
    def test_connect_edges_to_vertex_should_be_1(self):
        v1 = Vertex(0, 0)
        v2 = Vertex(1, 2)
        e = Edge(v1, v2)
        am.connect_edges_to_vertices([e])

        self.assertEqual(1, len([v for v in v1.connected_edges if v is not None]))

    def test_connect_edges_to_vertex_should_be_2(self):
        v1 = Vertex(0, 0)
        v2 = Vertex(1, 0)
        v3 = Vertex(-1, 0)
        e1 = Edge(v1, v2)
        e2 = Edge(v3, v1)
        am.connect_edges_to_vertices([e1, e2])
        self.assertEqual(2, len([v for v in v1.connected_edges if v is not None]))

    def test_connect_edges_to_vertex_should_be_sorted_right_left_up_down(self):
        v = Vertex(0, 0)
        e_left = Edge(Vertex(-1, 0), v)
        e_right = Edge(v, Vertex(1, 0))
        e_up = Edge(v, Vertex(0, 1))
        e_down = Edge(Vertex(0, -1), v)

        edges = [e_left, e_right, e_down, e_up]
        am.connect_edges_to_vertices(edges)
        ce = v.connected_edges
        self.assertEqual(ce[DIRECTION['RIGHT']], e_right)
        self.assertEqual(ce[DIRECTION['LEFT']], e_left)
        self.assertEqual(ce[DIRECTION['UP']], e_up)
        self.assertEqual(ce[DIRECTION['DOWN']], e_down)


if __name__ == '__main__':
    unittest.main()
