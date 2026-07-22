"""
End-to-end demo: horizontal rod clamped at the left end (vertices 0, 1
fixed), sagging under gravity with damping until it reaches equilibrium.

Run:  python demo_gravity.py
Output: demo_gravity.png (final shape, energy history, edge-length error).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rod

# ---- problem setup ----------------------------------------------------
n = 30                     # interior vertices
L = 1.0                    # rod length
alpha = 0.01               # bending modulus
beta = 0.01                # twist modulus (unused: stress-free frames)
rho = 1.0                  # mass per unit length
g = 9.8
h = 2e-4                   # time step
damping = 8.0
T = 4.0                    # total simulated time

s = np.linspace(0.0, L, n + 2)
x = np.stack([s, np.zeros_like(s), np.zeros_like(s)], axis=1)
rest_el = rod.rest_lengths(x)
vor_l = rod.voronoi_lengths(rest_el)
twoL = np.sum(vor_l)
mass = rod.vertex_masses(rest_el, rho)
v = np.zeros_like(x)
u0 = np.array([0.0, 1.0, 0.0])

fixed = np.zeros(n + 2, dtype=bool)
fixed[0] = fixed[1] = True           # clamp position and direction at left

gravity = np.zeros_like(x)
gravity[:, 2] = -g * mass
external = lambda y: gravity

# ---- run --------------------------------------------------------------
steps = int(T / h)
times, energies, cerrs = [], [], []
for k in range(steps):
    x, v, u0 = rod.step(x, v, u0, h, rest_el, vor_l, twoL, alpha, beta,
                        mass, fixed, external=external, damping=damping)
    if k % 200 == 0:
        times.append(k * h)
        energies.append(rod.bending_energy(x, rest_el, vor_l, alpha)
                        + np.sum(mass * g * x[:, 2]))
        cerrs.append(np.max(np.abs(rod.constraints(x, rest_el))))

print(f"final tip position       : {x[-1]}")
print(f"final speed (max)        : {np.max(np.linalg.norm(v, axis=1)):.2e}")
print(f"max edge-length error    : {max(cerrs):.2e}")

# ---- plots ------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(x[:, 0], x[:, 2], "o-", ms=3)
ax[0].set_aspect("equal"); ax[0].set_title("final shape (x-z)")
ax[1].plot(times, energies); ax[1].set_title("potential energy vs t")
ax[2].semilogy(times, np.maximum(cerrs, 1e-18))
ax[2].set_title("max |constraint| vs t")
for a in ax: a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("demo_gravity.png", dpi=130)
print("wrote demo_gravity.png")
