from typing import Callable

MathFunction = Callable[[float], float]


def trapezoidal_simple(f: MathFunction, a: float, b: float) -> float:
    return (b - a) / 2 * (f(a) + f(b))


def trapezoidal_composite(f: MathFunction, a: float, b: float, n: int) -> float:
    if n < 1:
        raise ValueError("N must be >= 1 for Trapezoidal Composite")

    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n):
        xi = a + i * h
        s += 2 * f(xi)

    return (h / 2) * s


def simpson_simple(f: MathFunction, a: float, b: float) -> float:
    m = (a + b) / 2
    return (b - a) / 6 * (f(a) + 4 * f(m) + f(b))


def simpson_composite(f: MathFunction, a: float, b: float, n: int) -> float:
    if n < 2 or (n % 2) != 0:
        raise ValueError("N must be even and >= 2 for Simpson Composite")

    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n):
        xi = a + i * h
        if i % 2 == 1:
            s += 4 * f(xi)
        else:
            s += 2 * f(xi)

    return (h / 3) * s
