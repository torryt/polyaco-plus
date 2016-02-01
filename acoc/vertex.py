class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connected_edges = []

    def __repr__(self):
        return "x: {:.3f}, y: {:.3f}".format(self.x, self.y)

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y))