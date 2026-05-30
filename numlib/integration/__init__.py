from .functions import MathFunction, EXAMPLE_FUNCTIONS, DEFAULT_EPSILONS
from .integrator import (
    IntegrationMethod,
    trapezoidal_simple,
    trapezoidal_composite,
    simpson_simple,
    simpson_composite,
)
from .integration_benchmarks import Manager

__all__ = [
    "MathFunction",
    "EXAMPLE_FUNCTIONS",
    "DEFAULT_EPSILONS",
    "IntegrationMethod",
    "trapezoidal_simple",
    "trapezoidal_composite",
    "simpson_simple",
    "simpson_composite",
    "Manager",
]
