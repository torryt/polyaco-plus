class BaseEdge:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return "a: ({}), b: ({})".format(self.a, self.b)

    def __eq__(self, other):
        if (self.a == other.a) and (self.b == other.b):
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.a, self.b))



class PolygonEdge(BaseEdge):
    def __init__(self, a, b):
        super(PolygonEdge, self).__init__(a, b)
        self.travel_count = 1


class MatrixEdge(BaseEdge):
    def __init__(self, a, b, pheromone_strength=0.1):
        super(MatrixEdge, self).__init__(a, b)
        self.pheromone_strength = pheromone_strength
