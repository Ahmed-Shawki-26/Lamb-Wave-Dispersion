"""
animate_lamb_wave.py
====================
Animated visualization of Lamb wave particle displacement for the
IM7/8552 CFRP quasi-isotropic plate at the 50 kHz excitation frequency.

Shows two modes side-by-side:
  - A0 (antisymmetric / flexural)  — the mode used in the Abaqus simulation
  - S0 (symmetric  / extensional)  — for comparison

Each panel is a quiver (arrow) plot showing how particles in the plate
cross-section move during one complete wave cycle.  Arrows = displacement
vectors; the pattern marches to the right as the wave propagates.

Outputs
-------
  lamb_wave_animation.gif   — animated GIF (Pillow writer, no extra installs)
  lamb_wave_animation.html  — interactive HTML5 animation (open in any browser)

Run
---
  PYTHONIOENCODING=utf-8 python animate_lamb_wave.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Path setup — same as dispersion_analysis.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# When run from inside this repo, script dir is the repo root (contains lambwaves/)
LWD_PATH   = SCRIPT_DIR
if LWD_PATH not in sys.path:
    sys.path.insert(0, LWD_PATH)

from lambwaves import Lamb

# ===========================================================================
# CFRP MATERIAL PARAMETERS  (identical to dispersion_analysis.py)
# ===========================================================================
Ex    = 61.758e9   # Pa   in-plane Young's modulus
nu_xy =  0.3188   #      in-plane Poisson's ratio
Gxz   =  4.466e9  # Pa   interlaminar shear modulus (governs c_S for Lamb)
rho   = 1622.0    # kg/m³
h_mm  = 24 * 0.132             # 3.168 mm  (24 plies × 0.132 mm)

c_S = np.sqrt(Gxz / rho)                                          # 1659 m/s
c_L = np.sqrt(Ex * (1-nu_xy) / (rho * (1+nu_xy) * (1-2*nu_xy))) # 7367 m/s

FREQ_KHZ  = 50.0
FD_EXCITE = FREQ_KHZ * h_mm    # 158.4  kHz·mm

print("=" * 60)
print("  Lamb Wave Animation  —  IM7/8552 CFRP  @  50 kHz")
print("=" * 60)
print(f"  c_L = {c_L:,.0f} m/s  |  c_S = {c_S:,.0f} m/s")
print(f"  h   = {h_mm:.3f} mm  |  fd  = {FD_EXCITE:.1f} kHz·mm")

# ===========================================================================
# SOLVE DISPERSION EQUATIONS
# fd_points=100 is enough — we only query at one fd value
# ===========================================================================
print("\n  Solving dispersion equations (reduced resolution for speed)...")
lamb = Lamb(
    thickness      = h_mm,
    nmodes_sym     = 2,
    nmodes_antisym = 2,
    fd_max         = 3000,
    vp_max         = 15000,
    c_L            = c_L,
    c_S            = c_S,
    fd_points      = 100,
    vp_step        = 50,
    material       = "IM7/8552",
)
print("  Done.\n")

# ===========================================================================
# QUERY VELOCITIES AT OPERATING POINT
# ===========================================================================
vp_a0 = float(lamb.vp_antisym["A0"](FD_EXCITE))
vp_s0 = float(lamb.vp_sym["S0"](FD_EXCITE))
freq_hz = FREQ_KHZ * 1e3

lambda_a0_mm = (vp_a0 / freq_hz) * 1e3   # spatial wavelength of A0 (mm)
lambda_s0_mm = (vp_s0 / freq_hz) * 1e3   # spatial wavelength of S0 (mm)

print(f"  A0  phase velocity = {vp_a0:>8.1f} m/s  |  wavelength = {lambda_a0_mm:.1f} mm")
print(f"  S0  phase velocity = {vp_s0:>8.1f} m/s  |  wavelength = {lambda_s0_mm:.1f} mm")

# ===========================================================================
# ANIMATION BUILDER
# Replicates the logic of Lamb.animate_displacement() but returns the
# FuncAnimation object so we can save with PillowWriter / HTMLWriter.
# ===========================================================================

def build_mode_animation_data(lamb_obj, mode_label, vp, fd):
    """
    Compute all data needed to animate one Lamb wave mode.

    Returns
    -------
    x, y        : 2-D meshgrids (m) — spatial domain
    time        : 1-D array of time samples (s) — one full wave period
    frames_u    : list of 2-D arrays — in-plane  displacement at each time step
    frames_w    : list of 2-D arrays — out-of-plane displacement at each time step
    max_disp    : float — maximum displacement magnitude (for scaling arrows)
    wavelength  : float (m)
    """
    d_m = lamb_obj.d           # plate thickness in metres
    h_m = lamb_obj.h           # half-thickness in metres

    freq   = fd / d_m          # frequency in Hz (fd is in kHz·mm, d in m → kHz/m → need care)
    # Note: the library stores d in metres (thickness_mm / 1000).
    # fd is in kHz·mm = (kHz)(mm). freq = fd / d = (kHz·mm) / m = kHz → ×1000 → Hz
    freq_hz  = freq * 1e3      # convert kHz to Hz
    omega    = 2 * np.pi * freq_hz
    k        = omega / vp      # wavenumber (rad/m)
    wavelength = vp / freq_hz  # spatial wavelength (m)

    # Spatial grid: one full wavelength in x, full thickness in y
    xx = np.linspace(0,  wavelength, 40)
    yy = np.linspace(-h_m, h_m, 40)
    x, y = np.meshgrid(xx, yy)

    # Time grid: one complete wave period
    T    = 1.0 / freq_hz       # period (s)
    time = np.linspace(0, T, 30)

    # Compute displacement frames
    frames_u, frames_w = [], []
    family = mode_label[0]     # 'A' or 'S'

    for t in time:
        u_struct, w_struct = lamb_obj._calc_wave_structure(family, vp, fd, y)
        u = u_struct * np.exp(1j * (k * x - omega * t))
        w = w_struct * np.exp(1j * (k * x - omega * t))
        frames_u.append(np.real(u))
        frames_w.append(np.real(w))

    max_disp = max(
        np.amax(np.sqrt(fu**2 + fw**2))
        for fu, fw in zip(frames_u, frames_w)
    )

    return x, y, time, frames_u, frames_w, max_disp, wavelength


print("\n  Computing displacement fields for A0 and S0...")

data_a0 = build_mode_animation_data(lamb, "A0", vp_a0, FD_EXCITE)
data_s0 = build_mode_animation_data(lamb, "S0", vp_s0, FD_EXCITE)

x_a0, y_a0, time_a0, fu_a0, fw_a0, maxd_a0, wl_a0 = data_a0
x_s0, y_s0, time_s0, fu_s0, fw_s0, maxd_s0, wl_s0 = data_s0

# ===========================================================================
# BUILD FIGURE WITH TWO SIDE-BY-SIDE PANELS
# ===========================================================================
fig, (ax_a0, ax_s0) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")

for ax in (ax_a0, ax_s0):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

def _setup_axis(ax, mode_label, vp, wavelength_m, lamb_obj, color):
    d_m = lamb_obj.d
    h_m = lamb_obj.h
    ax.set_title(f"Mode {mode_label}  |  vp = {vp:.0f} m/s  |  λ = {wavelength_m*1e3:.1f} mm",
                 fontsize=10, color="white")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_yticks([-h_m, 0, h_m])
    ax.set_yticklabels(["-d/2", "0", "+d/2"], color="white", fontsize=9)
    ax.set_ylabel("Plate thickness", color="white", fontsize=9)
    ax.set_xlim([-wavelength_m * 0.15, wavelength_m * 1.15])
    ax.set_ylim([-d_m * 0.7, d_m * 0.7])
    # Mid-plane reference line
    ax.axhline(0, color="#444444", lw=0.8, ls="--")
    # Plate boundary lines
    ax.axhline( h_m, color=color, lw=1.0, alpha=0.4)
    ax.axhline(-h_m, color=color, lw=1.0, alpha=0.4)

_setup_axis(ax_a0, "A0", vp_a0, wl_a0, lamb, "#ff6b6b")
_setup_axis(ax_s0, "S0", vp_s0, wl_s0, lamb, "#74b9ff")

fig.suptitle(
    "Lamb Wave Particle Displacement  —  IM7/8552 CFRP  @  50 kHz\n"
    "[45/-45/0/90]₃s   |   h = 3.168 mm   |   One full wave cycle",
    fontsize=11, color="white", y=1.01,
)

# Initial quiver plots (frame 0)
q_a0 = ax_a0.quiver(x_a0, y_a0, fu_a0[0], fw_a0[0],
                     scale=6 * maxd_a0, scale_units="inches",
                     color="#ff6b6b", alpha=0.85, width=0.003)

q_s0 = ax_s0.quiver(x_s0, y_s0, fu_s0[0], fw_s0[0],
                     scale=6 * maxd_s0, scale_units="inches",
                     color="#74b9ff", alpha=0.85, width=0.003)

# Frame counter text
txt = fig.text(0.5, -0.02,
               "Frame 1 / 30  —  t = 0.00 µs",
               ha="center", va="top", fontsize=8, color="#aaaaaa")

plt.tight_layout()

# ===========================================================================
# ANIMATION UPDATE FUNCTION
# ===========================================================================
n_frames = len(time_a0)

def update(frame_idx):
    q_a0.set_UVC(fu_a0[frame_idx], fw_a0[frame_idx])
    q_s0.set_UVC(fu_s0[frame_idx], fw_s0[frame_idx])
    t_us = time_a0[frame_idx] * 1e6
    txt.set_text(f"Frame {frame_idx+1} / {n_frames}  —  t = {t_us:.3f} µs")
    return q_a0, q_s0, txt

anim = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=60,       # ms between frames → ~0.5 s per loop
    blit=True,
    repeat=True,
)

# ===========================================================================
# SAVE OUTPUTS
# ===========================================================================
gif_path  = os.path.join(SCRIPT_DIR, "lamb_wave_animation.gif")
html_path = os.path.join(SCRIPT_DIR, "lamb_wave_animation.html")

print("\n  Saving GIF  (Pillow writer)  — this takes ~20 seconds...")
anim.save(gif_path, writer=animation.PillowWriter(fps=16),
          dpi=100, savefig_kwargs={"facecolor": "#0d1117"})
print(f"  GIF  saved  →  {gif_path}")
print(f"  Size         : {os.path.getsize(gif_path) / 1024:.0f} KB")

print("\n  Saving HTML (interactive browser animation)...")
anim.save(html_path, writer=animation.HTMLWriter(fps=16, embed_frames=True))
print(f"  HTML saved  →  {html_path}")
print(f"  Size         : {os.path.getsize(html_path) / 1024:.0f} KB")

# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 60)
print("  ANIMATION SUMMARY")
print("=" * 60)
print(f"  A0 (flexural)  : vp = {vp_a0:.0f} m/s  |  λ = {lambda_a0_mm:.1f} mm")
print(f"                   Displacement arrows rotate elliptically.")
print(f"                   Both in-plane (u) and out-of-plane (w) motion.")
print(f"                   This is the mode used in the Abaqus simulation.")
print()
print(f"  S0 (extensional): vp = {vp_s0:.0f} m/s  |  λ = {lambda_s0_mm:.1f} mm")
print(f"                   Arrows mostly horizontal (in-plane extension).")
print(f"                   Minimal out-of-plane motion at low fd.")
print()
print(f"  Output files:")
print(f"    {gif_path}")
print(f"    {html_path}")
print("=" * 60)
