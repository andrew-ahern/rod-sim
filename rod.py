"""
Discrete elastic rods (Bergou et al. 2008), specialized to naturally
straight, isotropic, uniform, inextensible open rods.

State:
    x  : (n+2, 3) vertex positions x_0 ... x_{n+1}
    v  : (n+2, 3) vertex velocities
    u0 : (3,)     Bishop frame vector at edge 0 (perpendicular to t^0)

Conventions follow the paper:
    edges   e^i = x_{i+1} - x_i,          i = 0..n      (n+1 edges)
    kb_i    curvature binormal, eq. (1),  i = 1..n      (interior vertices)
    l_i     = |e^{i-1}| + |e^i| (rest),   i = 1..n      (Voronoi factor)
    E_bend  = sum_i alpha |kb_i|^2 / l_i                (sec. 4.2.1, isotropic)
    E_twist = beta (theta^n - theta^0)^2 / (2L),  2L = sum_i l_i   (eq. 6)

All "rest_*" quantities are precomputed from the undeformed configuration
and held fixed (barred quantities in the paper).
"""

import numpy as np
from scipy.linalg import solveh_banded


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------

def edges(x):
    """e[i] = x[i+1] - x[i], shape (n+1, 3)."""
    return x[1:] - x[:-1]


def tangents(e):
    """Unit tangents per edge."""
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def rest_lengths(rest_x):
    """|ebar^i| per edge."""
    return np.linalg.norm(edges(rest_x), axis=1)


def voronoi_lengths(rest_el):
    """l_i = |ebar^{i-1}| + |ebar^i| for interior vertices i = 1..n."""
    return rest_el[:-1] + rest_el[1:]


def curvature_binormals(e, rest_el):
    """
    Integrated curvature binormal at interior vertices, eq. (1):
        kb_i = 2 e^{i-1} x e^i / (|ebar^{i-1}||ebar^i| + e^{i-1}.e^i)
    The product of edge lengths in the denominator uses rest values
    (edges are inextensible; this choice makes the gradient formulas
    of sec. 7.1 exact). Shape (n, 3), index 0 corresponds to vertex 1.
    """
    cross = np.cross(e[:-1], e[1:])
    denom = rest_el[:-1] * rest_el[1:] + np.einsum('ij,ij->i', e[:-1], e[1:])
    return 2.0 * cross / denom[:, None]


# ----------------------------------------------------------------------
# rotations and frames
# ----------------------------------------------------------------------

def rotate(vec, axis_unit, angle):
    """Rodrigues rotation of vec about unit axis by angle."""
    c, s = np.cos(angle), np.sin(angle)
    return (vec * c
            + np.cross(axis_unit, vec) * s
            + axis_unit * np.dot(axis_unit, vec) * (1.0 - c))


def transport(u, t_from, t_to):
    """
    Parallel transport u from tangent t_from to t_to: rotation about
    t_from x t_to taking t_from to t_to (identity if parallel).
    """
    axis = np.cross(t_from, t_to)
    na = np.linalg.norm(axis)
    if na < 1e-12:
        return u.copy()
    angle = np.arctan2(na, np.dot(t_from, t_to))
    return rotate(u, axis / na, angle)


def bishop_frames(t, u0):
    """
    Build Bishop frame vectors u^i along the rod by iterated parallel
    transport of u0 (sec. 4.2.2). Returns (n+1, 3) array of u vectors;
    v^i = t^i x u^i when needed.
    """
    n1 = len(t)
    u = np.empty((n1, 3))
    u[0] = u0
    for i in range(1, n1):
        u[i] = transport(u[i - 1], t[i - 1], t[i])
    return u


def signed_angle(a, b, t):
    """Signed angle from a to b about unit axis t (a, b perpendicular to t)."""
    return np.arctan2(np.dot(t, np.cross(a, b)), np.dot(a, b))


def unwrap_near(angle, ref):
    """Shift angle by a multiple of 2*pi to the branch nearest ref."""
    return angle + 2.0 * np.pi * np.round((ref - angle) / (2.0 * np.pi))


# ----------------------------------------------------------------------
# energies
# ----------------------------------------------------------------------

def bending_energy(x, rest_el, vor_l, alpha):
    e = edges(x)
    kb = curvature_binormals(e, rest_el)
    return alpha * np.sum(np.einsum('ij,ij->i', kb, kb) / vor_l)


def twist_energy(theta_diff, beta, twoL):
    """E_twist = beta * Theta^2 / (2L), with twoL = sum_i l_i (eq. 6)."""
    return beta * theta_diff ** 2 / twoL


# ----------------------------------------------------------------------
# forces
# ----------------------------------------------------------------------

def _skew(e):
    """[e] with [e] w = e x w."""
    return np.array([[0.0, -e[2], e[1]],
                     [e[2], 0.0, -e[0]],
                     [-e[1], e[0], 0.0]])


def bending_force(x, rest_el, vor_l, alpha):
    """
    F_i = - sum_{j=i-1}^{i+1} (2 alpha / l_j) (grad_i kb_j)^T kb_j,
    with the gradients of sec. 7.1:
        grad_{j-1} kb_j = (2[e^j]     + kb_j (e^j)^T    ) / D_j
        grad_{j+1} kb_j = (2[e^{j-1}] - kb_j (e^{j-1})^T) / D_j
        grad_j     kb_j = -(grad_{j-1} + grad_{j+1}) kb_j
        D_j = |ebar^{j-1}||ebar^j| + e^{j-1}.e^j
    """
    n2 = len(x)
    e = edges(x)
    kb = curvature_binormals(e, rest_el)          # kb[j-1] is kb at vertex j
    denom = rest_el[:-1] * rest_el[1:] + np.einsum('ij,ij->i', e[:-1], e[1:])
    F = np.zeros((n2, 3))
    n = n2 - 2
    for j in range(1, n + 1):                     # interior vertices
        k = kb[j - 1]
        D = denom[j - 1]
        gm = (2.0 * _skew(e[j]) + np.outer(k, e[j])) / D          # grad_{j-1}
        gp = (2.0 * _skew(e[j - 1]) - np.outer(k, e[j - 1])) / D  # grad_{j+1}
        g0 = -(gm + gp)                                           # grad_j
        c = 2.0 * alpha / vor_l[j - 1]
        F[j - 1] -= c * (gm.T @ k)
        F[j]     -= c * (g0.T @ k)
        F[j + 1] -= c * (gp.T @ k)
    return F


def holonomy_gradients(x, rest_el):
    """
    grad_i psi_j for the three nonzero vertex indices per interior
    vertex j (eq. 9):
        grad_{j-1} psi_j =  kb_j / (2 |ebar^{j-1}|)
        grad_{j+1} psi_j = -kb_j / (2 |ebar^j|)
        grad_j     psi_j = -(grad_{j-1} + grad_{j+1}) psi_j
    Returns arrays (gm, g0, gp), each (n, 3), row j-1 for vertex j.
    """
    e = edges(x)
    kb = curvature_binormals(e, rest_el)
    gm = kb / (2.0 * rest_el[:-1, None])
    gp = -kb / (2.0 * rest_el[1:, None])
    g0 = -(gm + gp)
    return gm, g0, gp


def twist_force(x, rest_el, theta_diff, beta, twoL):
    """
    F_i = + (beta Theta / L) sum_j grad_i psi_j   (sec. 7.1, special case),
    where beta Theta / L = dE_twist/dTheta = 2 beta Theta / (2L).
    """
    n2 = len(x)
    coef = 2.0 * beta * theta_diff / twoL         # = beta*Theta/L
    gm, g0, gp = holonomy_gradients(x, rest_el)
    F = np.zeros((n2, 3))
    n = n2 - 2
    for j in range(1, n + 1):
        F[j - 1] += coef * gm[j - 1]
        F[j]     += coef * g0[j - 1]
        F[j + 1] += coef * gp[j - 1]
    return F


# ----------------------------------------------------------------------
# inextensibility: fast projection (Goldenthal et al. 2007)
# ----------------------------------------------------------------------

def constraints(x, rest_el):
    """C_i = e^i.e^i - ebar^i.ebar^i per edge."""
    e = edges(x)
    return np.einsum('ij,ij->i', e, e) - rest_el ** 2


def fast_projection(x, rest_el, inv_mass, free_edge, tol=1e-10, max_iter=20):
    """
    Project x onto the constraint manifold C = 0, minimally in the
    mass metric. Each iteration solves the SPD tridiagonal system
        (grad C  M^-1  grad C^T) dlam = C,   x <- x - M^-1 grad C^T dlam.
    inv_mass: (n+2,) inverse vertex masses (0 for fixed vertices).
    free_edge: boolean (n+1,), False where both endpoints are fixed
    (those constraints are excluded; the BC maintains them).
    Returns projected x.
    """
    x = x.copy()
    idx = np.where(free_edge)[0]
    m = len(idx)
    if m == 0:
        return x
    for _ in range(max_iter):
        C = constraints(x, rest_el)[idx]
        if np.max(np.abs(C)) < tol:
            break
        e = edges(x)
        # grad C: row for edge i has -2e^i at vertex i, +2e^i at vertex i+1.
        # Assemble A = grad C M^-1 grad C^T restricted to free edges.
        diag = 4.0 * (inv_mass[idx] + inv_mass[idx + 1]) * \
            np.einsum('ij,ij->i', e[idx], e[idx])
        off = np.zeros(m - 1) if m > 1 else np.zeros(0)
        for a in range(m - 1):
            i, k = idx[a], idx[a + 1]
            if k == i + 1:  # adjacent edges share vertex i+1
                off[a] = -4.0 * inv_mass[i + 1] * np.dot(e[i], e[k])
        ab = np.zeros((2, m))
        ab[0, 1:] = off
        ab[1, :] = diag
        dlam = solveh_banded(ab, C)
        # x update: dx_p = -inv_mass_p * sum_i (dC_i/dx_p) dlam_i
        for a in range(m):
            i = idx[a]
            x[i] += inv_mass[i] * 2.0 * e[i] * dlam[a]
            x[i + 1] -= inv_mass[i + 1] * 2.0 * e[i] * dlam[a]
    return x


# ----------------------------------------------------------------------
# time stepping
# ----------------------------------------------------------------------

def vertex_masses(rest_el, mass_per_length):
    """Lumped vertex masses for a uniform rod."""
    n2 = len(rest_el) + 1
    m = np.zeros(n2)
    m[:-1] += 0.5 * mass_per_length * rest_el
    m[1:] += 0.5 * mass_per_length * rest_el
    return m


def step(x, v, u0, h, rest_el, vor_l, twoL, alpha, beta, mass, fixed,
         theta_diff=0.0, external=None, damping=0.0):
    """
    One time step: forces -> symplectic Euler -> fast projection ->
    velocity update -> time-parallel transport of u0.

    fixed      : boolean (n+2,), True where the vertex is prescribed
                 (its position is not updated here).
    theta_diff : imposed twist Theta = theta^n - theta^0 (0 => no twist
                 force; stress-free frames).
    external   : optional function external(x) -> (n+2,3) added force.
    damping    : viscous coefficient c, force -c m v.
    Returns updated (x, v, u0).
    """
    t0_old = edges(x)[0]
    t0_old = t0_old / np.linalg.norm(t0_old)

    F = bending_force(x, rest_el, vor_l, alpha)
    if theta_diff != 0.0:
        F += twist_force(x, rest_el, theta_diff, beta, twoL)
    if external is not None:
        F += external(x)
    if damping > 0.0:
        F -= damping * mass[:, None] * v

    free = ~fixed
    v_new = v.copy()
    v_new[free] += h * F[free] / mass[free, None]
    x_tent = x.copy()
    x_tent[free] += h * v_new[free]

    inv_mass = np.where(fixed, 0.0, 1.0 / mass)
    free_edge = ~(fixed[:-1] & fixed[1:])
    x_new = fast_projection(x_tent, rest_el, inv_mass, free_edge)

    v_new = (x_new - x) / h
    v_new[fixed] = v[fixed]

    t0_new = edges(x_new)[0]
    t0_new = t0_new / np.linalg.norm(t0_new)
    u0_new = transport(u0, t0_old, t0_new)
    # re-orthogonalize against drift
    u0_new -= np.dot(u0_new, t0_new) * t0_new
    u0_new /= np.linalg.norm(u0_new)

    return x_new, v_new, u0_new


# ----------------------------------------------------------------------
# clamped-frame bookkeeping (needed only for imposed twist)
# ----------------------------------------------------------------------

def frame_angles(x, u0, m1_clamp0, m1_clampn, theta0_prev, thetan_prev):
    """
    Given clamped material directions m1 at the end edges (fixed in
    space, perpendicular to the respective clamped tangents), return
    (theta0, thetan) on the branch nearest the previous values.
    """
    t = tangents(edges(x))
    u = bishop_frames(t, u0)
    th0 = unwrap_near(signed_angle(u[0], m1_clamp0, t[0]), theta0_prev)
    thn = unwrap_near(signed_angle(u[-1], m1_clampn, t[-1]), thetan_prev)
    return th0, thn
