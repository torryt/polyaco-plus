class Edge:
    def __init__(self, a, b, pheromone_strength=0.1):
        self.a = a
        self.b = b
        self.pheromone_strength = pheromone_strength

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