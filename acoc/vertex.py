

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connected_edges = [None] * 4

    def __repr__(self):
        return "x: {:.3f}, y: {:.3f}".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
        # return math.isclose(self.x, other.x, rel_tol=1e-4) and math.isclose(self.y, other.y, rel_tol=1e-4)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y))
