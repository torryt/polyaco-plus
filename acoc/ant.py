class Ant:
    def __init__(self, start_vertex):
        self.start_vertex = start_vertex
        self.current_edge = None
        self.edges_travelled = []
