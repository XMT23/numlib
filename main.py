"""
Mini visual tests written by Claude Code.
"""

import numpy as np

from numlib.ivp import *


def error_final(solver_gen, exact_fn):
    """Devuelve el error absoluto en el punto final."""
    t, y = None, None
    for t, y in solver_gen:
        pass
    y_exact = exact_fn(t)
    return np.abs(y - y_exact)


# ═══════════════════════════════════════════════════════
# 1. DECAIMIENTO EXPONENCIAL — y' = -y, y(0) = 1
#    Solución exacta: y(t) = e^(-t)
# ═══════════════════════════════════════════════════════
print("=" * 50)
print("1. Decaimiento exponencial: y' = -y")
print("=" * 50)

system1 = IVP(
    f=lambda t, v, p: -v.y,
    vars=["y"],
    t0=0.0,
    y0=1.0,
)
exact1 = lambda t: np.exp(-t)

tf, h = 5.0, 0.1
for nombre, gen in [
    ("Euler  ", euler(system1, tf, h)),
    ("Taylor2", taylor2(system1, tf, h)),
    ("RK2    ", rk2(system1, tf, h)),
    ("RK4    ", rk4(system1, tf, h)),
    ("AB2    ", ab2(system1, tf, h)),
    ("AB4    ", ab4(system1, tf, h)),
    ("AM     ", am(system1, tf, h)),
    ("PC    ", pc(system1, tf, h)),
]:
    err = error_final(gen, exact1)
    print(f"  {nombre}  error en t={tf}: {err[0]:.2e}")

# ═══════════════════════════════════════════════════════
# 2. CRECIMIENTO LOGÍSTICO — y' = r·y·(1 - y/K)
#    y(0) = y0, solución exacta: K / (1 + ((K-y0)/y0)·e^(-r·t))
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("2. Ecuación logística: y' = r·y·(1 - y/K)")
print("=" * 50)

r, K, y0_log = 1.0, 10.0, 1.0
system2 = IVP(
    f=lambda t, v, p: p.r * v.y * (1 - v.y / p.K),
    vars=["y"],
    t0=0.0,
    y0=y0_log,
    r=r,
    K=K,
)
exact2 = lambda t: K / (1 + ((K - y0_log) / y0_log) * np.exp(-r * t))

tf, h = 5.0, 0.1
for nombre, gen in [
    ("Euler  ", euler(system2, tf, h)),
    ("Taylor2", taylor2(system2, tf, h)),
    ("RK2    ", rk2(system2, tf, h)),
    ("RK4    ", rk4(system2, tf, h)),
    ("AB2    ", ab2(system2, tf, h)),
    ("AB4    ", ab4(system2, tf, h)),
    ("AM     ", am(system2, tf, h)),
    ("PC    ", pc(system2, tf, h)),
]:
    err = error_final(gen, exact2)
    print(f"  {nombre}  error en t={tf}: {err[0]:.2e}")

# ═══════════════════════════════════════════════════════
# 3. OSCILADOR ARMÓNICO — x'' + x = 0
#    Como sistema: x' = v, v' = -x
#    x(0)=1, v(0)=0 → x(t)=cos(t), v(t)=-sin(t)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("3. Oscilador armónico: x'' + x = 0")
print("=" * 50)

system3 = IVP(
    f=lambda t, v, p: [v.vx, -v.x],
    vars=["x", "vx"],
    t0=0.0,
    y0=[1.0, 0.0],
)
exact3 = lambda t: np.array([np.cos(t), -np.sin(t)])

tf, h = 2 * np.pi, 0.1  # una vuelta completa
for nombre, gen in [
    ("Euler  ", euler(system3, tf, h)),
    ("Taylor2", taylor2(system3, tf, h)),
    ("RK2    ", rk2(system3, tf, h)),
    ("RK4    ", rk4(system3, tf, h)),
    ("AB2    ", ab2(system3, tf, h)),
    ("AB4    ", ab4(system3, tf, h)),
    ("AM     ", am(system3, tf, h)),
    ("PC    ", pc(system3, tf, h)),
]:
    err = error_final(gen, exact3)
    print(f"  {nombre}  error en t=2π: x={err[0]:.2e}, v={err[1]:.2e}")


# ═══════════════════════════════════════════════════════
# 4. CONVERGENCIA — verificar que el error decrece con h
#    Usando decaimiento exponencial con RK4
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("4. Convergencia de RK4 al reducir h")
print("=" * 50)

for h in [0.5, 0.1, 0.05, 0.01]:
    gen = rk4(IVP(lambda t, v, p: -v.y, vars=["y"], t0=0.0, y0=1.0), tf=5.0, h=h)
    err = error_final(gen, lambda t: np.exp(-t))
    print(f"  h={h:.3f}  error: {err[0]:.2e}")


# ═══════════════════════════════════════════════════════
# 5. PASO VARIABLE — probar send()
#    h pequeño cuando |y| > 0.5, grande si no
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 50)
print("5. Paso variable con send()")
print("=" * 50)

system5 = IVP(
    f=lambda t, v, p: -v.y,
    vars=["y"],
    t0=0.0,
    y0=1.0,
)

gen = rk4(system5, tf=5.0, h=0.1)
t, y = next(gen)
pasos = 0
while True:
    new_h = 0.01 if abs(y[0]) > 0.5 else 0.2
    try:
        t, y = gen.send(new_h)
        pasos += 1
    except StopIteration:
        break

print(f"  Llegó a t={t:.4f}, y={y[0]:.6f}")
print(f"  Exacto:          y={np.exp(-t):.6f}")
print(f"  Error:             {abs(y[0] - np.exp(-t)):.2e}")
print(f"  Pasos realizados:  {pasos}")
