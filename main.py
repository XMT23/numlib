from typing import Tuple
from numlib.integration import *
from numlib.ivp import *

import numpy as np

EXAMPLE_IVP = {
    "exponencial": IVP(
        f=lambda t, v, p: -v.y,
        vars=["y"],
        t0=0.0,
        y0=1.0,
    ),
    "logistic": IVP(
        f=lambda t, v, p: p.r * v.y * (1 - v.y / p.K),
        vars=["y"],
        t0=0.0,
        y0=1.0,
        r=1.0,
        K=10.0,
    ),
    "oscil_harmonic": IVP(
        f=lambda t, v, p: [v.vx, -v.x],
        vars=["x", "vx"],
        t0=0.0,
        y0=[1.0, 0.0],
    )
}

EXAMPLE_IVP_EXACT = {
    "exponencial": lambda t: np.exp(-t),
    "logistic": lambda t: 10.0 / (1 + ((10.0 - 1.0) / 1.0) * np.exp(-1.0 * t)),
    "oscil_harmonic": lambda t: np.array([np.cos(t), -np.sin(t)]),
}

def error_final(solver_gen, exact_fn):
    t, y = None, None
    for t, y in solver_gen:
        pass
    y_exact = exact_fn(t)
    return np.abs(y - y_exact)

def get_user_input():
    func = None
    func_name = ""
    epsilon = 0.0
    a = 0.0
    b = 0.0
    max_n = 0

    while True:
        functions = []
        print("Selecciona una de les funcions disponibles:")
        for i, (k, v) in enumerate(EXAMPLE_FUNCTIONS.items()):
            print(f"{i+1}. {k}")
            functions.append((k, v))

        name, func = functions[int(input("> ")) - 1]

        epsilons = []
        print("Selecciona un valor per epsilon:")
        for i, (k, v) in enumerate(DEFAULT_EPSILONS.items()):
            print(f"{i+1}. {k}")
            epsilons.append(v)

        epsilon = epsilons[int(input("> ")) - 1]

        try:
            print("Introdueix el valor de n_max (enter per valor predeterminat - 100)")
            entrat = input("> ")
            if entrat.strip() == "":
                max_n = 100
            else:
                max_n = int(entrat)
                if max_n <= 0:
                    raise ValueError("n ha de ser major a 0")
            print("Introdueix l'extrem inferior de l'interval:")
            a = float(input("> "))

            print("Introdueix l'extrem superior de l'interval:")
            b = float(input("> "))
            break
        except ValueError as ve:
            print("valor introduit incorrecte:", ve)
        except Exception as e:
            print("hi ha agut un error:", e)

        print(f"reintentant...", end="\n\n")

    return func, name, epsilon, a, b, max_n


def select_ivp() -> Tuple[str, IVP]:
    selected_ivp = None

    while True:
        ivps = []
        print("Selecciona un dels problemes de valor inicial disponibles:")
        for i, (k, v) in enumerate(EXAMPLE_IVP.items()):
            print(f"{i+1}. {k}")
            ivps.append((k, v))

        try:
            name, ivp = ivps[int(input("> ")) - 1]
            return name, ivp
        except Exception as e:
            print("ERROR:", e)
            print("IVP seleccionat no disponible")

def select_h() -> float:
    while True:
        print("Introdueix un valor per h:")
        try:
            h = float(input("> "))
            if h <= 0:
                raise ValueError("h ha de ser un valor positiu")
            return h
        except Exception as e:
            print("ERROR:", e)

def select_tf() -> float:
    while True:
        print("Introdueix un valor per t final:")
        try:
            tf = float(input("> "))
            if tf <= 0:
                raise ValueError("t final ha de ser un valor positiu")
            return tf
        except Exception as e:
            print("ERROR:", e)

def solve_ivp(system: IVP, system_name: str,  tf: float, h: float):
    for name, gen in [
        ("Euler   ", euler(system, tf, h)),
        ("Taylor2 ", taylor2(system, tf, h)),
        ("RK2     ", rk2(system, tf, h)),
        ("RK4     ", rk4(system, tf, h)),
        ("AB2     ", ab2(system, tf, h)),
        ("AB4     ", ab4(system, tf, h)),
        ("AM      ", am(system, tf, h)),
        ("PC      ", pc(system, tf, h)),
    ]:
        err = error_final(gen, EXAMPLE_IVP_EXACT[system_name])
        print(f"{name} error in t={tf}: {err[0]:.2e}")

def main():
    print("=============================================")
    print("===== MÈTODES NUMERICS I PROBABILISTICS =====")
    print("=============================================")
    print("")

    while True:
        print("SELECCIONA FUNCIONALITAT:")
        print("1. Integració")
        print("2. Problema del valor inicial (EDO)")
        print("3. Sortir")
        try:
            seleccionat = int(input("> "))
            if seleccionat not in (1, 2, 3):
                raise ValueError(f"{seleccionat} no es una opcio valida")
            match seleccionat:
                case 1:
                    func, name, eps, a, b, max_n = get_user_input()
                    manager = Manager(func, name, eps)
                    table = manager.generate_table(a, b, max_n)
                    print(table)
                case 2:
                    system_name, system = select_ivp()
                    h = select_h()
                    tf = select_tf()
                    solve_ivp(system, system_name, tf, h)
                case 3:
                    exit()
        except Exception as e:
            print("ERROR:", e)

if __name__ == "__main__":
    main()
