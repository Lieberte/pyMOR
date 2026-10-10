"""Bounded-level pairwise accumulation using a backend's sparse matrices."""


class sparseAssembly:
    def __init__(self, shape, backend):
        self.backend = backend
        self.shape = shape
        self.levels = []

    def add(self, matrix):
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            matrix = self.levels[level] + matrix
            self.levels[level] = None
            level += 1
        if level == len(self.levels):
            self.levels.append(matrix)
        else:
            self.levels[level] = matrix

    def finish(self):
        result = None
        for matrix in self.levels:
            if matrix is not None:
                result = matrix if result is None else result + matrix
        self.levels.clear()
        return result if result is not None else self.backend.array.sparseFromTriplets([], [], [], self.shape)
