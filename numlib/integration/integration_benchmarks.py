from .functions import MathFunction
from .integrator import (
    IntegrationMethod,
    simpson_simple,
    simpson_composite,
    trapezoidal_simple,
    trapezoidal_composite,
)

import pandas as pd


class Manager:
    def __init__(
        self,
        function: MathFunction | None = None,
        function_name: str | None = None,
        epsilon: float | None = None,
    ):
        self.function = function
        self.function_name = function_name
        self.epsilon = epsilon
        self.methods: dict[str, IntegrationMethod] = {
            "Trap. Simple": trapezoidal_simple,
            "Simp. Simple": simpson_simple,
            "Trap. Compost": trapezoidal_composite,
            "Simp. Compost": simpson_composite,
        }

    def generate_table(self, a: float, b: float, max_n=1000) -> str:
        assert self.function is not None, "you must define f before analizying"
        assert self.epsilon is not None, "you must define epsilon before analizying"
        f = self.function

        results = []
        history: dict = {name: None for name in self.methods}
        converged = {name: False for name in self.methods}

        simple_values = {}
        for name, method in self.methods.items():
            if "Simple" in name:
                try:
                    val = method(f, a, b, 1)
                    simple_values[name] = f"{val:.8f}"
                except Exception as e:
                    simple_values[name] = e

        for n in range(1, max_n + 1):
            row = {"N": str(n)}
            all_converged = True

            for name, method in self.methods.items():
                if "Simp. Compost" in name and n % 2 == 1:
                    row[name] = "-"
                    all_converged = False
                    continue

                if "Simple" in name:
                    if True:
                        row[name] = simple_values[name]
                    else:
                        row[name] = "-"
                    continue

                try:
                    actual = method(f, a, b, n)
                except Exception as e:
                    print("error integrating", name, "- err:", e)
                    row[name] = "err"
                    all_converged = False
                    continue

                prev = history[name]
                row[name] = f"{actual:.8f}"

                if prev is not None and not converged[name]:
                    if abs(actual - prev) < self.epsilon:
                        converged[name] = True

                history[name] = actual

                if "Compost" in name and not converged[name]:
                    all_converged = False

            results.append(row)

            if all_converged:
                break

        df = pd.DataFrame(results)
        header = (
            f"Comparing methods for {self.function_name} (epsilon = {self.epsilon})"
        )
        return header + "\n" + df.to_string(index=False, justify="center")
