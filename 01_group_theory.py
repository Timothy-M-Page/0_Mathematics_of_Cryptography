import math
import itertools

import numpy as np

"""
A group G is a set of objects a,b,c... with a binary operation *
satisfying the four following axioms:

1. Closure: For all elements in G, a and b, a * b is also in G.

2. Associativity: a * ( b * c ) = ( a * b ) * c, for all a, b and c in G.

3. Identity element: There exists an element I in G such that,
                     for all a in G, I * a = a = a * I.

4. Inverse elements: For every element g in G, 
                     there exists an element h such that g*h = I.

Below are classes for the groups : additive and multiplicative integers
under a modulus, the dihedral group, the general linear group under 
a modulus and the symmetric group.
"""


class AdditiveGroupMod:
    """
    Finite group of integers less than or equal to a modulus.
    Every cyclic group is isomorphic to an additive group of
    integers under a modulus.
    """
    def __init__(self, n: int):
        self.n = n
        self.group = [x for x in range(0, n)]

    def add(self, x: int, y: int) -> int:
        return (x + y) % self.n

    def inverse(self, x: int) -> int:
        return (self.n - x) % self.n


class MultiplicativeGroupMod:
    """
    Finite group of integers coprime to a modulus, with the binary
    operation multiplication under the given modulus.

    The class contains element operations: products, inverses,
    exponentiation, the discrete logarithm.
    Group structure operations: subgroup, order and generators.
    """
    def __init__(self, mod: int):
        self.mod = mod
        # Form the set of integers less than and coprime to n, to form a group.
        self.group = [x for x in range(1, mod) if math.gcd(x, mod) == 1]

    def product(self, x: int, y: int) -> int:
        return (x * y) % self.mod

    def inverse(self, x: int) -> int:
        if math.gcd(x, self.mod) != 1:
            raise ValueError(f'{x} has no inverse mod {self.mod}.')
        return pow(x, -1, self.mod)

    def exponent(self, x: int, n: int) -> int:
        return pow(x, n, self.mod)

    def discrete_log(self, b: int, x: int) -> int:
        exponent = b % self.mod
        for i in range(1, len(self.group) + 1):
            if exponent == x:
                return i
            exponent = (exponent * b) % self.mod
        raise ValueError(
            f'There is no solution to {b}**n = {x} (mod {self.mod}).')

    def subgroup(self, x: int) -> list[int]:
        sub = []
        exponent = x
        for i in range(1, len(self.group)):
            if exponent not in sub:
                sub.append(exponent)
            exponent = (exponent * x) % self.mod
        return sub

    def order(self, x: int) -> int:
        return self.discrete_log(x, 1)

    def generators(self) -> list[int]:
        gen = []
        for g in self.group:
            if self.order(g) == self.mod - 1:
                gen.append(g)
        return gen


class DihedralGroup:
    """
    Dihedral group n is the set of symmetries on an n-sided regular polygon.
    This group has size 2n and is composed of n rotations and n reflections.
    It may be written isomorphically with the set of integers mod 2n, with
    rotations 0 to n-1 and reflections n through 2n - 1.
    """
    def __init__(self, n: int):
        self.n = n
        self.group = [x for x in range(0, (2 * n))]

    def product(self, x: int, y: int) -> int:
        if x < self.n and y < self.n:    # Rotations
            return (x + y) % self.n
        if x < self.n <= y:              # Rotation * Reflection
            return self.n + ((x + y) % self.n)
        if y < self.n <= x:              # Reflection * Rotation
            return self.n + ((x - y) % self.n)
        if x >= self.n and y >= self.n:  # Reflection * Reflection
            return (x - y) % self.n

    def inverse(self, x: int) -> int:
        if x >= self.n:
            return x
        return (self.n - x) % self.n


class GeneralLinearGroupMod:
    """
    The General Linear group is the set of matrices of dimension
    dim with integer entries mod n.
    """
    def __init__(self, dim: int, mod: int):
        self.mod = mod
        self.dim = dim

    def group(self) -> list:
        # Generate the set of all invertible matrices mod n.
        if self.dim == 1:
            return [np.array([i]) for i in range(1, self.mod)
                    if math.gcd(i, self.mod) == 1]
        matrices = []
        for elements in np.ndindex((self.mod,) * (self.dim * self.dim)):
            matrix = np.array(elements).reshape((self.dim, self.dim))
            det = round(np.linalg.det(matrix)) % self.mod
            if math.gcd(det, self.mod) == 1:
                matrices.append(matrix)
        return matrices

    def adjugate(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) == 1:
            return matrix
        rows, cols = matrix.shape
        cofactor_matrix = np.zeros_like(matrix, dtype=float)
        for i in range(rows):
            for j in range(cols):
                minor_matrix = np.delete(np.delete(matrix, i, axis=0), j,
                                         axis=1)
                cofactor_matrix[i, j] = ((-1) ** (i + j) *
                                         int(np.linalg.det(minor_matrix)
                                             % self.mod))
        return np.transpose(cofactor_matrix)

    def inverse(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) == 1:
            return np.array([pow(int(matrix[0]), -1, self.mod)])
        det = int(np.linalg.det(matrix) % self.mod)
        if det == 0:
            raise ValueError(f'Matrix determinant is zero.')
        det = det % self.mod
        if det == 0:
            raise ValueError(f'Matrix determinant is zero mod {self.mod}.')
        if math.gcd(det, self.mod) != 1:
            raise ValueError(f'Determinant is not coprime with modulus.')
        mod_det_inv = pow(det, -1, self.mod)
        adj = self.adjugate(matrix)
        inv_matrix = (mod_det_inv * adj) % self.mod
        return inv_matrix.astype(int)


class SymmetricGroup:
    """
    The Symmetric group size n is the set of permutations of the set
    [1, 2, ... n]. This group has n! elements and forms a group under
    the composition of permutations p1 * p2 (i) = p1(p2(i)). The group
    structure may be decomposed into cycles and transpositions that form
    a basis for all permutations.
    """
    def __init__(self, n: int):
        self.n = n
        self.group = [list(itertools.permutations(range(1, n + 1)))]

    @staticmethod
    def validate_permutation(perm: tuple) -> None:
        if set(perm) != set(range(len(perm))):
            raise ValueError(f'Permutation {perm} is invalid: it must be a '
                             f'tuple of integers from 0 to {len(perm)}.')

    def product(self, perm1: tuple[int], perm2: tuple[int]) -> tuple[int, ...]:
        self.validate_permutation(perm1)
        self.validate_permutation(perm2)
        product = tuple(perm1[perm2[i] - 1] for i in range(len(perm2)))
        return product

    def inverse(self, perm: tuple[int, ...]) -> tuple[int, ...]:
        self.validate_permutation(perm)
        n = len(perm)
        inverse = [0] * n
        for i, p in enumerate(perm):
            inverse[p - 1] = i + 1
        return tuple(inverse)

    def cycles(self, perm: tuple[int]) -> list[tuple[int, ...]]:
        # Decompose a permutation into disjoint cycles.
        self.validate_permutation(perm)
        visited = [False] * self.n
        cycles = []
        for i in range(self.n):
            if not visited[i]:
                cycle = []
                x = i
                while not visited[x]:
                    cycle.append(x + 1)  # Convert to 1-based indexing
                    visited[x] = True
                    x = perm[x] - 1  # Move to the next element in the cycle
                if len(cycle) > 1:  # Only cycles with more than one element
                    cycles.append(tuple(cycle))
        return cycles

    def transpositions(self, perm: tuple[int]) -> list[tuple[int, int]]:
        # Decompose a permutation into a product of transpositions.
        self.validate_permutation(perm)
        cycles = self.cycles(perm)
        transpositions = []
        for cycle in cycles:
            for i in range(len(cycle) - 1, 0, -1):
                transpositions.append((cycle[0], cycle[i]))
        return transpositions

    def sign(self, perm: tuple) -> int:
        # Compute the sign of a permutation.
        self.validate_permutation(perm)
        transpositions = self.transpositions(perm)
        return -1 if len(transpositions) % 2 else 1

    def adjacent_transpositions(self, perm: tuple) -> list[tuple[int, int]]:
        # Decompose a permutation into a product of adjacent transpositions.
        self.validate_permutation(perm)
        perm = list(perm)
        n = len(perm)
        transpositions = []
        # Bubblesort to sort the permutation using adjacent swaps
        for i in range(n):
            for j in range(n - 1):
                if perm[j] > perm[j + 1]:
                    perm[j], perm[j + 1] = perm[j + 1], perm[j]
                    transpositions.append((j + 1, j + 2))
        return transpositions
