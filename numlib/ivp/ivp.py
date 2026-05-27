import numpy as np

from typing import List, Callable


class IVP:
    def __init__(self, f, vars, t0, y0, **params) -> None:
        self.f: Callable = f
        self.t0: float = t0
        self.y0: np.ndarray = np.atleast_1d(y0)

        self._var_names: np.ndarray = np.array(vars)

        if len(self.y0) != len(self._var_names):
            raise ValueError(
                f"{len(self.y0)} expected variables, got {len(self._var_names)}"
            )

        self.params = type("Params", (), {})()
        for name, value in params.items():
            setattr(self.params, name, value)

    def evaluate(self, t: float | List[float], y: float | List[float] | np.ndarray):
        y_values = np.atleast_1d(y)

        if len(y_values) != len(self._var_names):
            raise ValueError(
                f"{len(self._var_names)} expected variables, got {len(y_values)}"
            )

        v = type("Vars", (), {})()
        for name, value in zip(self._var_names, y_values):
            setattr(v, name, value)

        result = self.f(t, v, self.params)
        return np.atleast_1d(result)
