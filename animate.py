"""
Animation utility: turn a recorded history of rod shapes into a video.

Usage in a driver:
    snaps = []
    for k in range(steps):
        x, v, u0 = rod.step(...)
        if k % record_every == 0:
            snaps.append(x.copy())
    animate.render(snaps, "run.mp4")            # or "run.gif"

render() writes .mp4 if ffmpeg is available, else fall back to .gif.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim


def render(snaps, filename, fps=30, view_3d=True, elev=20, azim=-60):
    """
    snaps: list of (n+2, 3) position arrays (equal time spacing).
    filename: .mp4 (requires ffmpeg) or .gif.
    view_3d: False gives an x-z plane view instead.
    """
    snaps = [np.asarray(s) for s in snaps]
    allpts = np.concatenate(snaps)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.05 * max(np.max(hi - lo), 1e-6)
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(6, 5))
    if view_3d:
        ax = fig.add_subplot(projection="3d")
        ax.view_init(elev=elev, azim=azim)
        line, = ax.plot([], [], [], "o-", ms=2.5, lw=1.5)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)

        def update(i):
            s = snaps[i]
            line.set_data(s[:, 0], s[:, 1])
            line.set_3d_properties(s[:, 2])
            return (line,)
    else:
        ax = fig.add_subplot()
        line, = ax.plot([], [], "o-", ms=2.5, lw=1.5)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2])
        ax.set_aspect("equal"); ax.grid(alpha=0.3)

        def update(i):
            s = snaps[i]
            line.set_data(s[:, 0], s[:, 2])
            return (line,)

    anim = manim.FuncAnimation(fig, update, frames=len(snaps), blit=True)
    if filename.endswith(".mp4") and "ffmpeg" in manim.writers.list():
        anim.save(filename, writer=manim.FFMpegWriter(fps=fps))
    else:
        if filename.endswith(".mp4"):
            filename = filename[:-4] + ".gif"
            print("ffmpeg not found; writing GIF instead")
        anim.save(filename, writer=manim.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {filename}")
