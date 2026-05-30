# numlib

A Python library implementing numerical methods for integration, interpolation,
and solving initial value problems (IVPs).
Developed as part of the *Numerical and Probabilistic Methods*
course at the Universitat Autònoma de Barcelona (UAB),
within the degree in Computational Mathematics and Data Analysis.

The library aims to be modular, readable, and easy to use. Each module is self-contained and exposes
a clean interface.

---

## Structure

```
.
├── main.py
└── numlib/
    ├── integration/
    │   ├── functions.py               # Example functions and epsilon presets
    │   ├── integrator.py              # Integration methods
    │   └── integration_benchmarks.py  # Convergence and comparision utilities
    ├── interpolation/
    │   └── interpolation.py           # Polynomial interpolation (Lagrange Polynomials) via Gaussian elimination
    └── ivp/
        ├── ivp.py                     # IVP problem definition class
        └── solver.py                  # All IVP solvers
```

**`numlib/ivp`** — Defines the `IVP` class for specifying ODE systems and
    implements 8 solvers ranging from Euler to a predictor-corrector scheme.
    All solvers are Python generators, which allows step-by-step iteration
    and dynamic step size control via `.send()`.

**`numlib/interpolation`** — Polynomial interpolation from a function and a set of nodes.
    The system is solved using a custom Gaussian elimination implementation.
    The result is an `InterpolationPoly` object that is callable and human-readable.

**`numlib/integration`** — Numerical integration using Trapezoidal and Simpson,
    with utilities to generate convergence tables across increasing subdivisions.

---

## Installation and requirements

It is recommended to create a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Examples

### Solving an IVP

Define the system using the `IVP` class.
Variables and parameters are accessed by name inside the function:

```python
from numlib.ivp import IVP, rk4

system = IVP(
    f=lambda t, v, p: -v.y,
    vars=["y"],
    t0=0.0,
    y0=1.0,
)

for t, y in rk4(system, tf=5.0, h=0.1):
    print(f"t={t:.2f}  y={y[0]:.6f}")
```

Parameters can be passed as keyword arguments and accessed via `p` inside the function:

```python
system = IVP(
    f=lambda t, v, p: p.r * v.y * (1 - v.y / p.K),
    vars=["y"],
    t0=0.0,
    y0=1.0,
    r=1.0,
    K=10.0,
)
```

Multi-variable systems work the same way.
Just provide a list of variable names and initial values:

```python
# Harmonic oscillator: x'' + x = 0  →  x' = vx, vx' = -x
system = IVP(
    f=lambda t, v, p: [v.vx, -v.x],
    vars=["x", "vx"],
    t0=0.0,
    y0=[1.0, 0.0],
)
```

---

### Variable step size with `.send()`

All solvers are generators.
You can send a new value of `h` at any step.
This might be useful for changing at runtime the stepsize.

```python
from numlib.ivp import IVP, rk4

system = IVP(f=lambda t, v, p: -v.y, vars=["y"], t0=0.0, y0=1.0)

gen = rk4(system, tf=5.0, h=0.1)
t, y = next(gen)

while True:
    new_h = 0.01 if abs(y[0]) > 0.5 else 0.2  # fine steps where y changes fast
    try:
        t, y = gen.send(new_h)
    except StopIteration:
        break
```

If no new step size is needed,
just iterate normally with `next()` or a `for` loop.

---

### Polynomial interpolation

Pass any callable and a list of interpolation nodes.
The result is an object you can call directly or print:

```python
from numlib.interpolation import interpolation_poly
import numpy as np

p = interpolation_poly(np.sin, [-1.0, 0.0, 1.0, 2.0])

print(p)        # → -0.1782x^3 + 0.0076x^2 + 0.8719x
print(p(0.5))   # → 0.4769...
```

The polynomial is solved internally via Gaussian elimination.

---

### Numerical integration

```python
from numlib.integration import simpson_composite

result = simpson_composite(lambda x: np.exp(-(x**2)), a=0.0, b=2.0, n=100)
print(result)  # → 0.8821...
```

---

## Implemented methods

### IVP Solvers (`numlib/ivp`)

| Method | Type | Order | Notes |
|--------|------|-------|-------|
| `euler` | Single-step | 1 | — |
| `taylor2` | Single-step | 2 | Jacobian estimated numerically if not provided |
| `rk2` | Single-step | 2 | Parameterizable via `alpha` |
| `rk4` | Single-step | 4 | — |
| `ab2` | Multi-step | 2 | Adams-Bashforth, starts with RK4 |
| `ab4` | Multi-step | 4 | Adams-Bashforth, starts with RK4 |
| `am` | Multi-step | 4 | Adams-Moulton (implicit) |
| `pc` | Multi-step | 4 | Predictor-corrector (AB4 + AM) |

### Integration methods (`numlib/integration`)

| Method | Type |
|--------|------|
| `trapezoidal_simple` | Simple rule |
| `trapezoidal_composite` | Composite rule |
| `simpson_simple` | Simple rule |
| `simpson_composite` | Composite rule |

### Interpolation (`numlib/interpolation`)

| Function | Description |
|----------|-------------|
| `interpolation_poly` | Builds interpolating polynomial from a callable and node list |
| `gauss` | Gaussian elimination for square linear systems |
| `triangular_sup` | Row reduction to upper triangular form |

---
