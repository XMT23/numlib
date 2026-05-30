from typing import Callable, List
import numpy as np


class InterpolationPoly:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def __str__(self) -> str:
        terms = []
        for i, a in enumerate(self.coefficients):
            if abs(a) < 1e-8:
                continue
            coef = f"{a:.4g}"
            if i == 0:
                terms.append(coef)
            elif i == 1:
                if str(coef) == "1":
                    coef = ""
                terms.append(f"{coef}x")
            else:
                if str(coef) == "1":
                    coef = ""
                terms.append(f"{coef}x^{i}")

        return " + ".join(list(reversed(terms))) if terms else "0"

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

    def evaluate(self, x: float) -> float:
        return np.sum([a * x**i for i, a in enumerate(self.coefficients)])


def _regressive_sustitution(sup_matrix: np.ndarray) -> np.ndarray:
    assert sup_matrix.shape[1] == sup_matrix.shape[0] + 1, (
        "'sup_matrix' does not represent a linear system of equations, "
        f"it must have n rows and n + 1 columns. Actual shape given: {sup_matrix.shape}"
    )

    coefficients = np.zeros_like(sup_matrix[0][:-1])

    for i in range(sup_matrix.shape[0] + 1):
        if sup_matrix[-i][-(i + 1)] == 0:
            coefficients[-i] = 0
        else:
            coefficients[-i] = (
                sup_matrix[-i][-1] - np.sum(coefficients * sup_matrix[-i][:-1])
            ) / sup_matrix[-i][-(i + 1)]

    return coefficients


def triangular_sup(matrix: np.ndarray) -> np.ndarray:
    if len(matrix.shape) != 2:
        raise ValueError(
            f"'matrix' must be a 2D array, got a {len(matrix.shape)}D object"
        )

    M = matrix.astype(float).copy()
    n_rows = M.shape[0]
    n_cols = M.shape[1]

    for i in range(n_rows):
        pivot = M[i][i]
        for k in range(i + 1, n_rows):
            m = M[k][i] / pivot
            for j in range(i, n_cols):
                M[k][j] = M[k][j] - M[i][j] * m

    return M


def gauss(M: np.ndarray) -> np.ndarray:
    if M.shape[0] > M.shape[1] - 1:
        raise ValueError("given system of equations doesn't have an unique solution")

    M = triangular_sup(M)

    return _regressive_sustitution(M)


def _coefficients(f: Callable, x: np.ndarray) -> np.ndarray:
    y = np.array([f(xi) for xi in x], dtype=float)
    m = len(x)

    A = np.array([[x[i] ** j for j in range(m)] for i in range(m)], dtype=float)
    M = np.hstack([A, y.reshape(-1, 1)])

    return gauss(M)


def interpolation_poly(
    f: Callable, x: List[float] | List[int] | np.ndarray
) -> InterpolationPoly:
    x_vec: np.ndarray = np.array(x, dtype=float)
    if len(x_vec) != len(np.unique(x_vec)):
        raise ValueError(f"vector 'x' must consist of unique elements. Got {x_vec}")

    coefficients: np.ndarray = _coefficients(f, x_vec)
    return InterpolationPoly(coefficients)
