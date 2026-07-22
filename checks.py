"""
Stage-1 gate: verify every analytic force against finite differences
of the energy, and verify fast projection restores edge lengths.

Run:  python checks.py
Pass: all reported max relative errors < 1e-6 and projection residual < 1e-10.
"""

import numpy as np
import rod

rng = np.random.default_rng(0)


def random_rod(n=8, seed=1):
    """Non-degenerate wiggly open rod with n interior vertices."""
    r = np.random.default_rng(seed)
    s = np.linspace(0.0, 1.0, n + 2)
    x = np.stack([s, 0.15 * np.sin(5 * s), 0.12 * np.cos(4 * s)], axis=1)
    x += 0.02 * r.standard_normal(x.shape)
    return x


def fd_gradient(fun, x, eps=1e-6):
    """Central-difference gradient of scalar fun(x), x shape (n+2,3)."""
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        for a in range(3):
            xp = x.copy(); xp[i, a] += eps
            xm = x.copy(); xm[i, a] -= eps
            g[i, a] = (fun(xp) - fun(xm)) / (2 * eps)
    return g


def rel_err(analytic, numeric):
    return np.max(np.abs(analytic - numeric)) / max(np.max(np.abs(numeric)), 1e-30)


# ----------------------------------------------------------------------
# 1. bending force
# ----------------------------------------------------------------------

def check_bending(alpha=1.3):
    x = random_rod()
    rest_x = random_rod(seed=2)          # rest shape irrelevant here except lengths
    rest_el = rod.rest_lengths(rest_x)
    vor_l = rod.voronoi_lengths(rest_el)
    F = rod.bending_force(x, rest_el, vor_l, alpha)
    G = fd_gradient(lambda y: rod.bending_energy(y, rest_el, vor_l, alpha), x)
    err = rel_err(F, -G)
    print(f"bending force  vs -dE/dx : max rel err = {err:.2e}")
    return err


# ----------------------------------------------------------------------
# 2. holonomy gradient (checks eq. 9 and its sign convention)
# ----------------------------------------------------------------------

def total_holonomy(x, x_ref, u0_ref):
    """
    Psi^n: angle by which the Bishop frame at the last edge rotates when
    the centerline moves x_ref -> x, with u0 time-parallel-transported.
    Measured against the tangent-transported old frame (sec. 6).
    """
    t_ref = rod.tangents(rod.edges(x_ref))
    u_ref = rod.bishop_frames(t_ref, u0_ref)
    t = rod.tangents(rod.edges(x))
    u0 = rod.transport(u0_ref, t_ref[0], t[0])
    u = rod.bishop_frames(t, u0)
    # transport old end-frame vector onto new end tangent, compare
    a = rod.transport(u_ref[-1], t_ref[-1], t[-1])
    return rod.signed_angle(a, u[-1], t[-1])


def check_holonomy():
    x = random_rod()
    rest_el = rod.rest_lengths(x)        # evaluate at the configuration itself
    u0 = np.array([0.0, 1.0, 0.0])
    t0 = rod.tangents(rod.edges(x))[0]
    u0 -= np.dot(u0, t0) * t0
    u0 /= np.linalg.norm(u0)

    gm, g0, gp = rod.holonomy_gradients(x, rest_el)
    n = len(gm)
    G = np.zeros_like(x)
    for j in range(1, n + 1):
        G[j - 1] += gm[j - 1]
        G[j]     += g0[j - 1]
        G[j + 1] += gp[j - 1]

    Gnum = fd_gradient(lambda y: total_holonomy(y, x, u0), x, eps=1e-6)
    err = rel_err(G, Gnum)
    print(f"holonomy grad  vs FD     : max rel err = {err:.2e}")
    return err


# ----------------------------------------------------------------------
# 3. total force with imposed twist (bending + twist, clamped frames)
# ----------------------------------------------------------------------

def check_total_with_twist(alpha=1.1, beta=0.7, Theta=3.0):
    x = random_rod()
    rest_el = rod.rest_lengths(x)
    vor_l = rod.voronoi_lengths(rest_el)
    twoL = np.sum(vor_l)
    u0 = np.array([0.0, 1.0, 0.0])
    t0 = rod.tangents(rod.edges(x))[0]
    u0 -= np.dot(u0, t0) * t0
    u0 /= np.linalg.norm(u0)

    def energy(y):
        # Theta(y) = Theta - Psi^n(y): Bishop frame at edge n rotates by
        # Psi^n, and the clamped material frame is fixed, so the angle
        # theta^n decreases by Psi^n (sec. 7.1).
        Th = Theta - total_holonomy(y, x, u0)
        return (rod.bending_energy(y, rest_el, vor_l, alpha)
                + rod.twist_energy(Th, beta, twoL))

    F = (rod.bending_force(x, rest_el, vor_l, alpha)
         + rod.twist_force(x, rest_el, Theta, beta, twoL))
    G = fd_gradient(energy, x, eps=1e-6)
    err = rel_err(F, -G)
    print(f"total force (twist)      : max rel err = {err:.2e}")
    return err


# ----------------------------------------------------------------------
# 4. fast projection
# ----------------------------------------------------------------------

def check_projection():
    x = random_rod()
    rest_el = rod.rest_lengths(x)
    mass = rod.vertex_masses(rest_el, 1.0)
    x_pert = x + 0.05 * rng.standard_normal(x.shape)   # violates constraints
    fixed = np.zeros(len(x), dtype=bool)
    fixed[0] = fixed[1] = True
    inv_mass = np.where(fixed, 0.0, 1.0 / mass)
    free_edge = ~(fixed[:-1] & fixed[1:])
    # keep the fixed edge at rest length (BC responsibility)
    x_pert[0], x_pert[1] = x[0], x[1]
    x_proj = rod.fast_projection(x_pert, rest_el, inv_mass, free_edge)
    res = np.max(np.abs(rod.constraints(x_proj, rest_el)))
    print(f"projection residual      : {res:.2e}")
    print(f"fixed vertices moved     : {np.max(np.abs(x_proj[:2]-x[:2])):.2e}")
    return res


if __name__ == "__main__":
    e1 = check_bending()
    e2 = check_holonomy()
    e3 = check_total_with_twist()
    r = check_projection()
    ok = e1 < 1e-6 and e2 < 1e-6 and e3 < 1e-6 and r < 1e-10
    print("\nGATE:", "PASS" if ok else "FAIL")
