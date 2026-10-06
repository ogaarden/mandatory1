"""Create report/neumannwave.gif for task 1.2.6.

Neumann problem with mx = my = 2 and CFL = 1/sqrt(2).
Run from the repository root:  python make_gif.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from Wave2D import Wave2D_Neumann

N = 40
mx = my = 2
cfl = 1 / np.sqrt(2)
c = 1.0

# One period of the standing wave: T = 2*pi/w, w = c*pi*sqrt(mx^2 + my^2)
w = c * np.pi * np.sqrt(mx**2 + my**2)
dt = cfl * (1 / N) / c
steps_per_period = int(round(2 * np.pi / w / dt))
Nt = steps_per_period  # one full period, so the loop is seamless

sol = Wave2D_Neumann()
data = sol(N, Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=1)
xij, yij = sol.create_mesh(N)

fig = plt.figure(figsize=(5, 4.5), dpi=80)
ax = fig.add_subplot(projection="3d")
frames = []
for n in sorted(data):
    if n % 2 or n == Nt:  # every second step; skip last (= first) frame
        continue
    surf = ax.plot_surface(
        xij, yij, data[n], cmap="viridis", vmin=-1, vmax=1,
        linewidth=0, antialiased=False, rstride=1, cstride=1,
    )
    title = ax.text2D(0.5, 0.95, f"t = {n * dt:.3f}",
                      transform=ax.transAxes, ha="center")
    frames.append([surf, title])

ax.set_zlim(-1.05, 1.05)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
ax.set_title("Neumann, $m_x=m_y=2$, $C=1/\\sqrt{2}$", pad=0)

anim = animation.ArtistAnimation(fig, frames, interval=60, blit=True, repeat_delay=0)
os.makedirs("report", exist_ok=True)
out = os.path.join("report", "neumannwave.gif")
anim.save(out, writer=animation.PillowWriter(fps=15))
print(f"Saved {out}: {os.path.getsize(out) / 1e6:.2f} MB, {len(frames)} frames")