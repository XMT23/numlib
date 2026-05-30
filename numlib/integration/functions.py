from typing import Callable, Dict
import numpy as np

MathFunction = Callable[[float], float]

EXAMPLE_FUNCTIONS: Dict[str, MathFunction] = {
    "x²": lambda x: x**2,
    "x³ + 2x² - 20": lambda x: x**3 + 2 * x**2 - 20,
    "5x⁴ - 3x² + 2": lambda x: 5 * x**4 - 3 * x**2 + 2,
    "sin(x)": lambda x: np.sin(x),
    "cos(x) - (x² - 1/x)": lambda x: np.cos(x) - (x**2 - 1 / x),
    "sin(x²)": lambda x: np.sin(x**2),
    "2x / (3 + √x)": lambda x: (2 * x) / (3 + np.sqrt(x)),
    "1 / (1 + x²)": lambda x: 1 / (1 + x**2),
    "√(1 + x³)": lambda x: np.sqrt(1 + x**3),
    "e^x": lambda x: np.exp(x),
    "e^(-x²)": lambda x: np.exp(-(x**2)),
    "ln(x + 1)": lambda x: np.log(x + 1),
}

DEFAULT_EPSILONS: Dict[str, float] = {
    "10^-2": 1e-2,
    "10^-5": 1e-5,
    "10^-8": 1e-8,
    "10^-12": 1e-12,
}
