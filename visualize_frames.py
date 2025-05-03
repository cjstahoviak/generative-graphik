"""
visualize_se3_chain.py
Visualises a chain of randomly‑perturbed SE(3) transforms and draws the
x‑/y‑/z‑axes of every frame in 3‑D.
-----------------------------------------------------------------------
Install deps (Python ≥3.8):
    pip install numpy matplotlib liegroups
Run:
    python visualize_se3_chain.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – side‑effect: 3‑D backend
from liegroups import SE3, SO3                   # pip install liegroups


# ----------------------------------------------------------------------
# Low‑level helper: draw one coordinate frame
# ----------------------------------------------------------------------
def draw_frame(ax, T: SE3, label: str = "", length: float = 0.05) -> None:
    """
    Draw the three unit axes of an SE(3) transform `T` as coloured arrows.

    Args
    ----
    ax     : 3‑D matplotlib axis.
    T      : SE3 instance whose .rot (SO3) and .trans (3,) we use.
    label  : Optional text tag at the frame origin.
    length : Arrow length in metres.
    """
    origin = T.trans                          # (3,)
    R = T.rot.as_matrix()                     # (3,3) – convert SO3 → ndarray

    colors = ['r', 'g', 'b']                  # x, y, z
    for i in range(3):
        ax.quiver(origin[0], origin[1], origin[2],     # start
                  *(R[:, i] * length),                 # direction
                  color=colors[i],
                  arrow_length_ratio=0.15,
                  linewidth=1.5)

    if label:
        ax.text(origin[0], origin[1], origin[2], label, fontsize=8)


# ----------------------------------------------------------------------
# Mid‑level helper: draw many frames in one figure
# ----------------------------------------------------------------------
def visualize_transforms(T_list, title="SE(3) transforms", lim=0.2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, T in enumerate(T_list):
        draw_frame(ax, T, label=f"p{i}")

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])              # equal aspect in Matplotlib ≥3.3
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Demo / test: build a kinematic chain with random perturbations
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    T_chain = [SE3.identity()]
    T_running = SE3.identity()

    for _ in range(6):
        # --- random translation (±5 cm per axis) ----------------------
        t_delta = 0.1 * (np.random.rand(3) - 0.5)

        # --- random rotation (≤ ±0.4 π rad about random axis) ---------
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = 0.4 * np.pi * (2 * np.random.rand() - 1)
        R_delta = SO3.exp(axis * angle).as_matrix()

        # --- assemble small SE(3) perturbation ------------------------
        T_delta = SE3.from_matrix(np.block([
            [R_delta, t_delta.reshape(3, 1)],
            [np.zeros((1, 3)), 1],
        ]))

        # --- compose with running transform --------------------------
        T_running = T_running.dot(T_delta)
        T_chain.append(T_running)

    visualize_transforms(T_chain,
                         title="Perturbed SE(3) link frames (non‑coplanar)")
