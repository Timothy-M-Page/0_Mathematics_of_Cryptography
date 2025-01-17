import math

import numpy as np


def determinant(matrix: np.ndarray) -> int:
    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix must be a numpy array.')
    det = np.linalg.det(matrix)
    # Avoid floating point precision errors.
    if math.isclose(det, round(det)):
        return round(det)
    else:
        return det


def transpose(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix must be a numpy array.')
    return np.transpose(matrix)


def rank(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f'Matrix must be a numpy array.')
    result = np.linalg.matrix_rank(matrix)
    return result


def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix must be a numpy array.')
    if np.linalg.det(matrix) == 0:
        raise ValueError(f'Matrix has no inverse due to determinant = 0.')
    return np.linalg.inv(matrix)


def adjugate(matrix: np.ndarray) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix must be a numpy array.')
    rows, cols = matrix.shape
    cofactor_matrix = np.zeros_like(matrix, dtype=float)
    for i in range(rows):
        for j in range(cols):
            minor_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            cofactor_matrix[i, j] = (-1)**(i+j) * determinant(minor_matrix)
    return np.transpose(cofactor_matrix)


def mod_matrix_inverse(matrix: np.ndarray, mod: int) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError('Matrix must be a numpy array.')
    if not isinstance(mod, int):
        raise TypeError('Modulus must be an integer.')
    det = determinant(matrix)
    if det == 0:
        raise ValueError('Matrix determinant is zero.')
    det = det % mod
    if det == 0:
        raise ValueError(f'Matrix determinant is zero mod {mod}.')
    if math.gcd(det, mod) != 1:
        raise ValueError(f'Determinant must be coprime with modulus.')
    mod_det_inv = pow(det, -1, mod)
    adj = adjugate(matrix)
    inv_matrix = (mod_det_inv * adj) % mod
    return inv_matrix.astype(int)
