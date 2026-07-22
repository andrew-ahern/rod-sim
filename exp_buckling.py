"""
Stage 2: Euler buckling of a cantilever (clamped-free rod) under an
axial compressive tip load P.

Theory: the straight state is stable for P < Pc = pi^2 alpha / (4 L^2)
and buckles above. We give the rod a small lateral perturbation, run
damped dynamics, and classify the straight state as stable (perturbation
decays) or unstable (grows). Bisection on P locates the threshold;
we report it for increasing n to show convergence to theory.

Run:  python exp_buckling.py     (a few minutes)
"""

import numpy as np
import rod

alpha = 0.01
beta = 0.01
L = 1.0
rho = 1.0
Pc_theory = np.pi ** 2 * alpha / (4 * L ** 2)


def setup(n):
    s = np.linspace(0.0, L, n + 2)
    x = np.stack([s, np.zeros_like(s), np.zeros_like(s)], axis=1)
    rest_el = rod.rest_lengths(x)
    vor_l = rod.voronoi_lengths(rest_el)
    mass = rod.vertex_masses(rest_el, rho)
    fixed = np.zeros(n + 2, dtype=bool)
    fixed[0] = fixed[1] = True
    return s, x, rest_el, vor_l, np.sum(vor_l), mass, fixed


def perturbation_growth(n, P, T=6.0, h=None, a0=1e-3, damping=1.0):
    """Return A(T)/A(T/2), lateral amplitude ratio (>1 => unstable)."""
    s, x, rest_el, vor_l, twoL, mass, fixed = setup(n)
    if h is None:
        ds = L / (n + 1)
        h = 0.3 * ds ** 2 * np.sqrt(rho / alpha)   # bending CFL-type limit
    x[:, 1] += a0 * (1 - np.cos(np.pi * s / (2 * L)))  # first-mode-like bow
    x[0, 1] = x[1, 1] = 0.0
    v = np.zeros_like(x)
    u0 = np.array([0.0, 1.0, 0.0])

    tip_load = np.zeros_like(x)
    tip_load[-1, 0] = -P
    external = lambda y: tip_load

    steps = int(T / h)
    half = steps // 2
    A_half = A_end = None
    for k in range(steps):
        x, v, u0 = rod.step(x, v, u0, h, rest_el, vor_l, twoL, alpha, beta,
                            mass, fixed, external=external, damping=damping)
        if k == half:
            A_half = np.max(np.abs(x[:, 1]))
        A_end = np.max(np.abs(x[:, 1]))
    return A_end / A_half


def find_threshold(n, lo=0.5 * Pc_theory, hi=2.0 * Pc_theory, iters=8):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if perturbation_growth(n, mid) > 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print(f"theory: Pc = {Pc_theory:.5f}\n")
    for n in (10, 20, 40):
        Pc = find_threshold(n)
        print(f"n = {n:3d} : Pc = {Pc:.5f}   rel err = {abs(Pc-Pc_theory)/Pc_theory:.3%}")
