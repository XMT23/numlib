from typing import Callable, Optional, Tuple, Generator
from .ivp import IVP
import numpy as np

def euler_step(
    system: IVP, t: float, y: np.ndarray, h: float
) -> Tuple[float, np.ndarray]:
    dy = system.evaluate(t, y)
    return t + h, y + h * dy

def taylor2_step(
    system: IVP,
    t: float,
    y: np.ndarray,
    h: float,
    df_dy: Optional[Callable] = None,
    df_dt: Callable | None = None,
    eps: float = 1e-6
) -> Tuple[float, np.ndarray]:
    f = system.evaluate(t, y)

    if df_dy is not None:
        d1 = df_dy(t, y)
    else:
        n = len(y)
        d1 = np.zeros((n,n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = eps
            d1[:, i] = (system.evaluate(t, y + e) - system.evaluate(t, y - e)) / (2*eps)

    if df_dt is not None:
        d2 = df_dt(t, y)
    else:
        d2 = (system.evaluate(t + eps, y) - system.evaluate(t - eps, y)) / (2*eps)

    return t + h, y + h*f + (h**2/2) * (d1 @ f + d2)

def rk2_step(
    system: IVP, t: float, y: np.ndarray, h: float, alpha: float = 0.5
) -> Tuple[float, np.ndarray]:
    c1, c2 = 1 - alpha, alpha
    a2 = 1 / (2 * alpha)
    b21 = 1 / (2 * alpha)

    k1 = system.evaluate(t, y)
    k2 = system.evaluate(t + h * a2, y + h * b21 * k1)

    return t + h, y + h * (c1 * k1 + c2 * k2)

def rk4_step(system: IVP, t: float, y: np.ndarray, h: float) -> Tuple[float, np.ndarray]:
    k1 = system.evaluate(t, y)
    k2 = system.evaluate(t + h/2, y + (h/2) * k1)
    k3 = system.evaluate(t + h/2, y + (h/2) * k2)
    k4 = system.evaluate(t + h,   y + h * k3)

    return t + h, y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


def _solver(step_fn: Callable, system: IVP, tf: float, h: float, **kwargs):
    t, y = system.t0, system.y0.copy()
    while t < tf:
        h_actual = min(h, tf - t)
        h_nuevo = yield t, y
        if h_nuevo is not None:
            h = h_nuevo
        t, y = step_fn(system, t, y, h_actual, **kwargs)
    yield t, y


def euler(system: IVP, tf: float, h: float) -> Generator:
    return _solver(euler_step, system, tf, h)

def taylor2(
    system: IVP,
    tf: float,
    h: float,
    df_dy: Callable | None = None,
    df_dt: Callable | None = None,
    eps: float = 1e-6
) -> Generator:
    return _solver(taylor2_step, system, tf, h, df_dy=df_dy, df_dt=df_dt, eps=eps)

def euler2(system: IVP, tf: float, h: float):
    return _solver(rk2_step, system, tf, h, alpha=1.0)

def rk2(system: IVP, tf: float, h: float, alpha: float = 0.5) -> Generator:
    return _solver(rk2_step, system, tf, h, alpha=alpha)

def rk4(system: IVP, tf: float, h: float) -> Generator:
    return _solver(rk4_step, system, tf, h)

