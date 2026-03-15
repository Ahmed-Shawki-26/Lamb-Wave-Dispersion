"""
lamb_wave_gui.py
================
Desktop GUI for the Lamb-Wave-Dispersion library.
All outputs in one window — no coding required.

Tabs:
  1  Setup          — material preset / CLT / solver params / SOLVE
  2  Dispersion     — phase velocity, group velocity, wave number
  3  Wave Structure — thickness-wise displacement profiles
  4  Animation      — quiver displacement animation (GIF + HTML)
  5  Export         — save TXT results, open output folder

Run:
  PYTHONIOENCODING=utf-8 python lamb_wave_gui.py
"""

import sys
import os
import shutil
import threading
import webbrowser
import numpy as np

# Matplotlib — Agg backend so plt.show() is a no-op (must be before pyplot)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser


# ---------------------------------------------------------------------------
# Tooltip helper  — shows a pop-up explanation when hovering over any widget
# ---------------------------------------------------------------------------
class ToolTip:
    """Lightweight tooltip that appears after a short delay on hover."""

    PAD = 6   # pixels of internal padding

    def __init__(self, widget, text, delay=500):
        self._widget = widget
        self._text   = text
        self._delay  = delay          # ms before the tip appears
        self._job    = None
        self._win    = None
        widget.bind("<Enter>",    self._schedule, add="+")
        widget.bind("<Leave>",    self._cancel,   add="+")
        widget.bind("<Button>",   self._cancel,   add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._job = self._widget.after(self._delay, self._show)

    def _cancel(self, _event=None):
        if self._job:
            self._widget.after_cancel(self._job)
            self._job = None
        self._hide()

    def _show(self):
        if self._win:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw, text=self._text, justify="left",
            background="#2d2d2d", foreground="#f0f0f0",
            relief="flat", borderwidth=1,
            font=("Segoe UI", 9),
            wraplength=420,
            padx=self.PAD, pady=self.PAD,
        )
        lbl.pack()

    def _hide(self):
        if self._win:
            self._win.destroy()
            self._win = None


# ---------------------------------------------------------------------------
# Tooltip text for every input parameter
# ---------------------------------------------------------------------------
TIPS = {
    # ---- Ply / material properties ----------------------------------------
    "E1": (
        "Longitudinal (fibre-direction) Young's modulus  [GPa]\n\n"
        "How to find it:\n"
        "  • Look up the data sheet for your prepreg system (e.g. T300/5208).\n"
        "  • Typical CFRP range: 100 – 180 GPa.\n"
        "  • NASA RP-1351 and Soden et al. (1998) tabulate values for common systems.\n"
        "  • For a quick estimate from rule-of-mixtures:  E1 ≈ Vf·Ef + (1-Vf)·Em"
    ),
    "E2": (
        "Transverse Young's modulus (perpendicular to fibres)  [GPa]\n\n"
        "How to find it:\n"
        "  • From the same prepreg data sheet.\n"
        "  • Typical CFRP range: 8 – 15 GPa.\n"
        "  • Much lower than E1 because it is matrix-dominated."
    ),
    "G12": (
        "In-plane shear modulus  [GPa]\n\n"
        "How to find it:\n"
        "  • From the prepreg data sheet (sometimes listed as G_LT or G_12).\n"
        "  • Typical CFRP range: 4 – 8 GPa.\n"
        "  • Matrix-dominated property."
    ),
    "nu12": (
        "Major Poisson's ratio  (dimensionless)\n\n"
        "How to find it:\n"
        "  • From the prepreg data sheet (listed as ν_12 or ν_LT).\n"
        "  • Typical CFRP range: 0.20 – 0.35.\n"
        "  • The minor Poisson's ratio ν21 = ν12 × E2/E1  (computed internally)."
    ),
    "rho": (
        "Material density  [kg/m³]\n\n"
        "How to find it:\n"
        "  • From the prepreg data sheet (cured-ply density).\n"
        "  • Typical CFRP range: 1 450 – 1 600 kg/m³.\n"
        "  • Can also be measured:  mass / volume  of a cured coupon."
    ),
    "thickness": (
        "Total laminate thickness  [mm]\n\n"
        "How to find it:\n"
        "  • Measure with digital calipers on the actual plate.\n"
        "  • Or calculate:  number_of_plies × cured_ply_thickness.\n"
        "    (Typical cured ply thickness ≈ 0.125 – 0.25 mm.)\n\n"
        "Why it matters:\n"
        "  The dispersion curves are plotted against f·d (kHz·mm), so every\n"
        "  velocity value depends on this thickness."
    ),
    # ---- Wave velocities ---------------------------------------------------
    "cL": (
        "Longitudinal (bulk) wave velocity  [m/s]\n\n"
        "What it is:\n"
        "  Speed of a compressive wave through the equivalent isotropic laminate.\n"
        "  Computed automatically by the CLT button using the quasi-isotropic\n"
        "  laminate invariants.\n\n"
        "How to override:\n"
        "  Type a value directly if you know it from ultrasonic measurements\n"
        "  or a more detailed model.  Typical CFRP range: 5 000 – 10 000 m/s.\n\n"
        "Role in the solver:\n"
        "  Sets the upper search bound for phase velocity and the CFL time step."
    ),
    "cS": (
        "Shear (bulk) wave velocity  [m/s]\n\n"
        "What it is:\n"
        "  Speed of a transverse wave through the equivalent isotropic laminate.\n"
        "  Computed from G_xy and ρ via CLT.\n\n"
        "Typical CFRP range: 2 000 – 4 000 m/s."
    ),
    "cR": (
        "Rayleigh / surface wave velocity  [m/s]  (optional)\n\n"
        "What it is:\n"
        "  Approximate speed of a surface wave; shown as a reference line on\n"
        "  dispersion plots.  Leave blank to skip.\n\n"
        "Estimated from c_S:  c_R ≈ c_S × (0.862 + 1.14·ν) / (1 + ν).\n"
        "The CLT button computes this automatically."
    ),
    # ---- Solver settings ---------------------------------------------------
    "nmodes_sym": (
        "Number of symmetric Lamb wave modes to compute  (S0, S1, S2 …)\n\n"
        "Guidance:\n"
        "  • 5 modes is sufficient for most SHM / NDE studies.\n"
        "  • Increase if your fd_max is large and you need higher modes.\n"
        "  • More modes = longer solve time."
    ),
    "nmodes_antisym": (
        "Number of antisymmetric Lamb wave modes to compute  (A0, A1, A2 …)\n\n"
        "Guidance:\n"
        "  • 5 is sufficient for most SHM studies.\n"
        "  • A0 is the most commonly used mode in low-frequency SHM.\n"
        "  • More modes = longer solve time."
    ),
    "fd_max": (
        "Maximum frequency × thickness product  [kHz·mm]\n\n"
        "What it is:\n"
        "  The solver sweeps from 0 to fd_max.  The dispersion curves are\n"
        "  plotted on this axis.\n\n"
        "How to choose:\n"
        "  fd_max = your maximum test frequency (kHz) × plate thickness (mm).\n"
        "  Example: 500 kHz on a 2 mm plate  →  fd_max = 1 000 kHz·mm.\n"
        "  The default (5 000) covers most lab setups."
    ),
    "vp_max": (
        "Maximum phase velocity to search  [m/s]\n\n"
        "What it is:\n"
        "  The solver will not look for modes with vp above this value.\n\n"
        "Guidance:\n"
        "  Keep it above c_L (the longitudinal bulk speed).  15 000 m/s is a\n"
        "  safe default for CFRP."
    ),
    "fd_points": (
        "Number of frequency–thickness grid points  (resolution of the curves)\n\n"
        "Guidance:\n"
        "  • 200 gives smooth curves for most purposes.\n"
        "  • Increase to 500–1000 for publication-quality figures.\n"
        "  • More points = longer solve time."
    ),
    "vp_step": (
        "Phase velocity search step size  [m/s]\n\n"
        "What it is:\n"
        "  The solver scans phase velocity in steps of this size looking for\n"
        "  sign changes in the Rayleigh–Lamb characteristic equation.\n\n"
        "Guidance:\n"
        "  • 50 m/s is a good default.\n"
        "  • Decrease to 10–20 if modes are missing or cutoff frequencies are wrong.\n"
        "  • Smaller step = longer solve time."
    ),
    "mat_name": (
        "Material label  (free text)\n\n"
        "Used as:\n"
        "  • The title on all saved plots and text files.\n"
        "  • Part of the output file name  (e.g. T300_5208_plate_1mm.txt).\n\n"
        "Leave blank for 'custom'."
    ),
    # ---- Abaqus inputs -----------------------------------------------------
    "freq": (
        "Excitation frequency  [kHz]\n\n"
        "What it is:\n"
        "  The centre frequency of the tone-burst signal applied to the actuator.\n\n"
        "How to choose:\n"
        "  • Look at the dispersion curves: pick an fd value where the mode of\n"
        "    interest has low dispersion (flat group-velocity curve).\n"
        "  • fd (kHz·mm) = freq (kHz) × thickness (mm).\n"
        "  • Common SHM range: 50 – 500 kHz for 1 – 4 mm CFRP plates."
    ),
    "epw": (
        "Elements per wavelength  (spatial resolution criterion)\n\n"
        "What it is:\n"
        "  Minimum number of elements that must fit across one wavelength.\n"
        "  Determines the in-plane element size:  Le = λ / epw.\n\n"
        "Guidance:\n"
        "  • Use ≥ 10 for engineering accuracy (rule of thumb).\n"
        "  • Use ≥ 20 for high-fidelity research simulations.\n"
        "  • More elements per wavelength → finer mesh → more computation."
    ),
    "eth": (
        "Elements through the plate thickness  (depth resolution)\n\n"
        "Guidance:\n"
        "  • Minimum 2 for A0 / S0 modes.\n"
        "  • Use 4–8 for higher-order modes or accurate through-thickness stress.\n"
        "  • For a 1 mm plate with eth=2:  each element is 0.5 mm thick."
    ),
    "cfl": (
        "CFL (Courant–Friedrichs–Lewy) safety factor  (time-step control)\n\n"
        "What it is:\n"
        "  The stable time step is:  dt = CFL × Le / (c_L × √3).\n"
        "  A factor < 1 keeps the explicit solver stable.\n\n"
        "Guidance:\n"
        "  • 0.9 is the standard engineering choice (10 % below the stability limit).\n"
        "  • Lower values (0.5–0.8) give extra safety for complex geometry.\n"
        "  • Never use > 1.0 — the simulation will diverge."
    ),
    "cyc": (
        "Number of cycles in the tone-burst excitation signal\n\n"
        "What it is:\n"
        "  A Hanning-windowed tone burst of N cycles is applied to the actuator.\n\n"
        "How to choose:\n"
        "  • More cycles → narrower frequency bandwidth (better mode selection).\n"
        "  • Fewer cycles → shorter pulse (better spatial resolution).\n"
        "  • 3–5 cycles is a common SHM choice.\n"
        "  • Signal duration:  t1 = N / f_c."
    ),
    "amp": (
        "Peak displacement amplitude of the excitation  [mm]\n\n"
        "What it is:\n"
        "  Applied as a U2 (out-of-plane) displacement boundary condition on the\n"
        "  actuator node set.\n\n"
        "Guidance:\n"
        "  • Keep small (0.01 – 0.1 mm) to stay in the linear elastic regime.\n"
        "  • Larger amplitude → bigger signal but may trigger material nonlinearity."
    ),
    "tsim": (
        "Total simulation time  [µs]\n\n"
        "How to choose:\n"
        "  • Must be long enough for the wave packet to travel across the plate\n"
        "    and for reflections to arrive at the sensor.\n"
        "  • Estimate:  t_sim > 2 × plate_length / group_velocity.\n"
        "  • Example: 300 mm plate, vg = 3 000 m/s → t > 200 µs → use 300 µs.\n\n"
        "Longer time → more time steps → higher computation cost."
    ),
    "pL": (
        "Plate length  (X direction)  [mm]\n\n"
        "Guidance:\n"
        "  • Match your experimental specimen dimensions.\n"
        "  • Larger plate → more elements → higher computation cost.\n"
        "  • Consider using absorbing boundaries / infinite elements if you only\n"
        "    need the direct wave (avoids reflections without a huge plate)."
    ),
    "pW": (
        "Plate width  (Y direction)  [mm]\n\n"
        "Same guidance as Plate L.  For a square plate set L = W."
    ),
}


def add_tip(widget, key):
    """Attach a ToolTip to *widget* using the text in TIPS[key]."""
    if key in TIPS:
        ToolTip(widget, TIPS[key])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# When run from inside this repo, script dir is the repo root (contains lambwaves/)
LWD_PATH    = SCRIPT_DIR
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

def _safe_folder_name(mat_name, thickness, c_L=None):
    """Build a filesystem-safe folder name: Material_thicknessMM[_cLXXXX]."""
    mat = (mat_name or "custom").strip()
    thick = (thickness or "0").strip()
    safe = mat.replace("/", "_").replace(" ", "_").strip() or "custom"
    parts = [safe, f"{thick}mm"]
    if c_L:
        try:
            parts.append(f"cL{int(float(c_L))}")
        except (ValueError, TypeError):
            pass
    return "_".join(parts)

if LWD_PATH not in sys.path:
    sys.path.insert(0, LWD_PATH)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.chdir(SCRIPT_DIR)   # library saves to ./results/ relative to CWD

from lambwaves import Lamb   # noqa: E402  (import after path setup)

# ---------------------------------------------------------------------------
# Material presets  (from PROJECT_GUIDE.md §3)
# ---------------------------------------------------------------------------
PRESETS = {
    "T300/5208":  {"E1": 132.0, "E2": 10.8, "G12": 5.65, "nu12": 0.24, "rho": 1520, "thickness": 1.0},
    "T300/914":   {"E1": 138.0, "E2": 11.0, "G12": 5.5,  "nu12": 0.28, "rho": 1550, "thickness": 1.0},
    "AS4/3501-6": {"E1": 148.0, "E2": 10.5, "G12": 5.61, "nu12": 0.30, "rho": 1540, "thickness": 1.0},
}

# ---------------------------------------------------------------------------
# CLT — Laminate Invariant Method for [0/45/-45/90]s  (PROJECT_GUIDE §14.3)
# ---------------------------------------------------------------------------
def clt_velocities(E1_GPa, E2_GPa, G12_GPa, nu12, rho):
    E1 = E1_GPa * 1e9;  E2 = E2_GPa * 1e9;  G12 = G12_GPa * 1e9
    nu21 = nu12 * E2 / E1
    D    = 1.0 - nu12 * nu21
    Q11  = E1 / D;  Q22 = E2 / D;  Q12 = nu12 * E2 / D;  Q66 = G12
    U1   = (3*Q11 + 3*Q22 + 2*Q12 + 4*Q66) / 8.0
    U4   = (  Q11 +   Q22 + 6*Q12 - 4*Q66) / 8.0
    Ex   = (U1**2 - U4**2) / U1
    nxy  = U4 / U1
    Gxy  = (U1 - U4) / 2.0
    c_L  = np.sqrt(Ex * (1 - nxy) / (rho * (1 + nxy) * (1 - 2*nxy)))
    c_S  = np.sqrt(Gxy / rho)
    c_R  = c_S * (0.862 + 1.14 * nxy) / (1.0 + nxy)
    return round(c_L), round(c_S), round(c_R)

# ---------------------------------------------------------------------------
# Animation data builder  (reused from animate_lamb_wave.py)
# ---------------------------------------------------------------------------
def build_anim_data(lamb_obj, mode, vp, fd):
    d = lamb_obj.d;  h = lamb_obj.h
    freq_hz = (fd / d) * 1e3
    omega   = 2.0 * np.pi * freq_hz
    k       = omega / vp
    wl      = vp / freq_hz
    xx = np.linspace(0,  wl, 40)
    yy = np.linspace(-h, h,  40)
    x, y = np.meshgrid(xx, yy)
    T     = 1.0 / freq_hz
    times = np.linspace(0, T, 30)
    family = mode[0]
    fu, fw = [], []
    for t in times:
        us, ws = lamb_obj._calc_wave_structure(family, vp, fd, y)
        fu.append(np.real(us * np.exp(1j * (k*x - omega*t))))
        fw.append(np.real(ws * np.exp(1j * (k*x - omega*t))))
    maxd = max(np.amax(np.sqrt(u**2 + w**2)) for u, w in zip(fu, fw))
    return x, y, times, fu, fw, maxd, wl


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lamb Wave Dispersion  —  CFRP Toolkit")
        self.geometry("1150x820")
        self.minsize(920, 660)

        self.lamb       = None      # Lamb object, set after SOLVE
        self._sym_col   = "#1f77b4"
        self._anti_col  = "#d62728"
        self._figs      = {}        # {key: fig} — track open figures for cleanup
        self._abq_params = None     # computed Abaqus parameters dict

        self._build_ui()

    def get_material_output_dir(self):
        """Return path to material-specific output folder (created if needed)."""
        mat   = self._sv["mat_name"].get().strip() if hasattr(self, "_sv") else ""
        thick = self._pv["thickness"].get().strip() if hasattr(self, "_pv") else "0"
        c_L   = self._vv["cL"].get().strip() if hasattr(self, "_vv") else ""
        name  = _safe_folder_name(mat, thick, c_L)
        path  = os.path.join(RESULTS_DIR, name)
        os.makedirs(path, exist_ok=True)
        return path

    # -----------------------------------------------------------------------
    # UI skeleton
    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_setup = ttk.Frame(self.nb); self.nb.add(self.tab_setup, text="  Setup  ")
        self.tab_disp  = ttk.Frame(self.nb); self.nb.add(self.tab_disp,  text="  Dispersion  ")
        self.tab_wstr  = ttk.Frame(self.nb); self.nb.add(self.tab_wstr,  text="  Wave Structure  ")
        self.tab_anim  = ttk.Frame(self.nb); self.nb.add(self.tab_anim,  text="  Animation  ")
        self.tab_abq   = ttk.Frame(self.nb); self.nb.add(self.tab_abq,   text="  Abaqus Parameters  ")
        self.tab_exp   = ttk.Frame(self.nb); self.nb.add(self.tab_exp,   text="  Export  ")
        self.tab_help  = ttk.Frame(self.nb); self.nb.add(self.tab_help,  text="  Help / Guide  ")

        self._build_setup()
        self._build_dispersion()
        self._build_wave_structure()
        self._build_animation()
        self._build_abaqus()
        self._build_export()
        self._build_help()

    # =======================================================================
    # TAB 1 — SETUP
    # =======================================================================
    def _build_setup(self):
        p = self.tab_setup

        # Preset buttons
        pf = ttk.LabelFrame(p, text="Quick Presets  (optional — or type any material values directly below)")
        pf.pack(fill="x", padx=10, pady=(10, 4))
        for name in PRESETS:
            ttk.Button(pf, text=name,
                       command=lambda n=name: self._load_preset(n)
                       ).pack(side="left", padx=10, pady=8)
        ttk.Button(pf, text="Clear / New Material",
                   command=self._clear_fields).pack(side="left", padx=16, pady=8)
        ttk.Label(pf, text="All fields below are freely editable — enter any material you like.",
                  foreground="#555", font=("", 8)).pack(side="left", padx=8)

        # Two-column body
        cols = ttk.Frame(p)
        cols.pack(fill="both", expand=True, padx=10, pady=4)

        left  = ttk.LabelFrame(cols, text="Ply Properties  +  Wave Velocities  [0/45/-45/90]s")
        right = ttk.LabelFrame(cols, text="Solver Settings")
        left.pack(side="left",  fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        # --- Left: ply properties ---
        prop_defs = [
            ("E1   (GPa)  (?)",      "e1",        "E1"),
            ("E2   (GPa)  (?)",      "e2",        "E2"),
            ("G12  (GPa)  (?)",      "g12",       "G12"),
            ("ν12          (?)",     "nu12",      "nu12"),
            ("ρ    (kg/m³)  (?)",    "rho",       "rho"),
            ("Thickness  (mm)  (?)", "thickness", "thickness"),
        ]
        self._pv = {}
        for r, (lbl, key, tip_key) in enumerate(prop_defs):
            lw = ttk.Label(left, text=lbl, foreground="#1a73e8", cursor="question_arrow")
            lw.grid(row=r, column=0, sticky="w", padx=10, pady=5)
            add_tip(lw, tip_key)
            v = tk.StringVar()
            ttk.Entry(left, textvariable=v, width=14).grid(row=r, column=1, padx=10, pady=5)
            self._pv[key] = v

        ttk.Separator(left, orient="horizontal").grid(
            row=len(prop_defs), column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Button(left, text="Compute  c_L / c_S / c_R  (CLT)",
                   command=self._compute_vel
                   ).grid(row=len(prop_defs)+1, column=0, columnspan=2, padx=10, pady=4)

        vel_defs = [
            ("c_L  (m/s)  (?)", "cL", "cL"),
            ("c_S  (m/s)  (?)", "cS", "cS"),
            ("c_R  (m/s)  (?)", "cR", "cR"),
        ]
        self._vv = {}
        base = len(prop_defs) + 2
        for i, (lbl, key, tip_key) in enumerate(vel_defs):
            lw = ttk.Label(left, text=lbl, foreground="#1a73e8", cursor="question_arrow")
            lw.grid(row=base+i, column=0, sticky="w", padx=10, pady=5)
            add_tip(lw, tip_key)
            v = tk.StringVar()
            ttk.Entry(left, textvariable=v, width=14).grid(row=base+i, column=1, padx=10, pady=5)
            self._vv[key] = v
        ttk.Label(left, text="(c_R is optional — leave blank to skip)",
                  foreground="#888", font=("", 8)
                  ).grid(row=base+3, column=0, columnspan=2, padx=10, sticky="w")

        # --- Right: solver settings ---
        solver_defs = [
            ("Symmetric modes  (?)",    "nmodes_sym",     "5",     "nmodes_sym"),
            ("Antisymm. modes  (?)",    "nmodes_antisym", "5",     "nmodes_antisym"),
            ("fd_max   (kHz·mm)  (?)",  "fd_max",         "5000",  "fd_max"),
            ("vp_max   (m/s)  (?)",     "vp_max",         "15000", "vp_max"),
            ("fd_points  (?)",          "fd_points",      "200",   "fd_points"),
            ("vp_step  (m/s)  (?)",     "vp_step",        "50",    "vp_step"),
            ("Material name  (?)",      "mat_name",       "",      "mat_name"),
        ]
        self._sv = {}
        for r, (lbl, key, dflt, tip_key) in enumerate(solver_defs):
            lw = ttk.Label(right, text=lbl, foreground="#1a73e8", cursor="question_arrow")
            lw.grid(row=r, column=0, sticky="w", padx=10, pady=5)
            add_tip(lw, tip_key)
            v = tk.StringVar(value=dflt)
            ttk.Entry(right, textvariable=v, width=16).grid(row=r, column=1, padx=10, pady=5)
            self._sv[key] = v

        ttk.Label(right, text="\nTip: fd_max = max_frequency × thickness.\nFor 50 kHz on a 1 mm plate → fd = 50.\n5000 gives plenty of modes for analysis.",
                  foreground="#666", font=("", 8), justify="left"
                  ).grid(row=len(solver_defs), column=0, columnspan=2, padx=10, sticky="w")

        # Solve row
        bf = ttk.Frame(p)
        bf.pack(fill="x", padx=10, pady=(6, 12))
        self._solve_btn = ttk.Button(bf, text="   SOLVE / UPDATE   ", command=self._start_solve)
        self._solve_btn.pack(side="left", padx=8, ipady=6)
        self._status = tk.StringVar(value="Not solved yet.  Fill in material properties + velocities, then click SOLVE.")
        ttk.Label(bf, textvariable=self._status, foreground="#444").pack(side="left", padx=8)

    # =======================================================================
    # TAB 2 — DISPERSION CURVES
    # =======================================================================
    def _build_dispersion(self):
        p = self.tab_disp

        ctrl = ttk.Frame(p)
        ctrl.pack(fill="x", padx=10, pady=(8, 4))

        # Plot type
        bf = ttk.LabelFrame(ctrl, text="Plot Type")
        bf.pack(side="left", padx=(0, 8))
        ttk.Button(bf, text="Phase Velocity",
                   command=self._plot_phase).pack(side="left", padx=4, pady=4)
        ttk.Button(bf, text="Group Velocity",
                   command=self._plot_group).pack(side="left", padx=4, pady=4)
        ttk.Button(bf, text="Wave Number",
                   command=self._plot_wnum).pack(side="left", padx=4, pady=4)

        # Modes
        mf = ttk.LabelFrame(ctrl, text="Modes")
        mf.pack(side="left", padx=(0, 8))
        self._d_modes = tk.StringVar(value="both")
        for val, lbl in [("both", "Both"), ("symmetric", "Sym only"), ("antisymmetric", "Anti only")]:
            ttk.Radiobutton(mf, text=lbl, variable=self._d_modes,
                            value=val).pack(side="left", padx=4, pady=4)

        # Options
        of = ttk.LabelFrame(ctrl, text="Options")
        of.pack(side="left", padx=(0, 8))
        self._d_cutoff = tk.BooleanVar(value=True)
        self._d_matvel = tk.BooleanVar(value=True)
        self._d_save   = tk.BooleanVar(value=False)
        ttk.Checkbutton(of, text="Cutoff freqs",  variable=self._d_cutoff).pack(side="left", padx=4)
        ttk.Checkbutton(of, text="Material vel.", variable=self._d_matvel).pack(side="left", padx=4)
        ttk.Checkbutton(of, text="Save PNG",      variable=self._d_save).pack(side="left", padx=4)

        # Colors
        cf = ttk.LabelFrame(ctrl, text="Curve Colors")
        cf.pack(side="left", padx=(0, 8))
        self._sym_btn  = ttk.Button(cf, text=f"Sym:  {self._sym_col}",
                                     command=lambda: self._pick_color("sym"))
        self._anti_btn = ttk.Button(cf, text=f"Anti: {self._anti_col}",
                                     command=lambda: self._pick_color("anti"))
        self._sym_btn.pack(side="left", padx=4, pady=4)
        self._anti_btn.pack(side="left", padx=4, pady=4)

        # Canvas area
        self._df = ttk.Frame(p)
        self._df.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        ttk.Label(self._df, text="Solve first (Setup tab), then click a plot type above.",
                  foreground="#888").pack(expand=True)

    # =======================================================================
    # TAB 3 — WAVE STRUCTURE
    # =======================================================================
    def _build_wave_structure(self):
        p = self.tab_wstr

        ctrl = ttk.Frame(p)
        ctrl.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Label(ctrl, text="Mode:").pack(side="left")
        self._ws_mode = tk.StringVar(value="A0")
        self._ws_mode_cb = ttk.Combobox(ctrl, textvariable=self._ws_mode,
                                         values=["A0"], width=7, state="readonly")
        self._ws_mode_cb.pack(side="left", padx=(2, 14))

        ttk.Label(ctrl, text="Rows:").pack(side="left")
        self._ws_nrows = tk.StringVar(value="2")
        ttk.Spinbox(ctrl, textvariable=self._ws_nrows, from_=1, to=6,
                    width=4).pack(side="left", padx=(2, 14))

        ttk.Label(ctrl, text="Cols:").pack(side="left")
        self._ws_ncols = tk.StringVar(value="3")
        ttk.Spinbox(ctrl, textvariable=self._ws_ncols, from_=1, to=6,
                    width=4).pack(side="left", padx=(2, 14))

        ttk.Label(ctrl, text="fd values (\"auto\"  or  comma-sep, e.g. 50,100,200,500,1000,2000):").pack(side="left")
        self._ws_fd = tk.StringVar(value="auto")
        ttk.Entry(ctrl, textvariable=self._ws_fd, width=36).pack(side="left", padx=(2, 14))

        self._ws_save = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Save PNG", variable=self._ws_save).pack(side="left", padx=(0, 8))
        ttk.Button(ctrl, text="Plot", command=self._plot_wave_struct).pack(side="left", padx=4)

        self._wsf = ttk.Frame(p)
        self._wsf.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        ttk.Label(self._wsf, text="Solve first, then select a mode and click Plot.",
                  foreground="#888").pack(expand=True)

    # =======================================================================
    # TAB 4 — ANIMATION
    # =======================================================================
    def _build_animation(self):
        p = self.tab_anim

        cf = ttk.LabelFrame(p, text="Animation Parameters")
        cf.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Label(cf, text="Mode:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self._a_mode = tk.StringVar(value="A0")
        self._a_mode_cb = ttk.Combobox(cf, textvariable=self._a_mode,
                                        values=["A0"], width=8, state="readonly")
        self._a_mode_cb.grid(row=0, column=1, padx=8)

        ttk.Label(cf, text="fd  (kHz·mm):").grid(row=0, column=2, sticky="w", padx=8)
        self._a_fd = tk.StringVar(value="50")
        ttk.Entry(cf, textvariable=self._a_fd, width=10).grid(row=0, column=3, padx=8)

        ttk.Label(cf, text="Frame interval  (ms):").grid(row=0, column=4, sticky="w", padx=8)
        self._a_speed     = tk.IntVar(value=60)
        self._a_speed_lbl = tk.StringVar(value="60 ms")
        ttk.Scale(cf, variable=self._a_speed, from_=20, to=150, orient="horizontal",
                  length=130,
                  command=lambda v: self._a_speed_lbl.set(f"{int(float(v))} ms")
                  ).grid(row=0, column=5, padx=8)
        ttk.Label(cf, textvariable=self._a_speed_lbl, width=6).grid(row=0, column=6, padx=4)

        self._anim_btn = ttk.Button(cf, text="  Generate & Open in Browser  ",
                                     command=self._gen_anim)
        self._anim_btn.grid(row=1, column=0, columnspan=7, pady=(4, 10), padx=10)

        ttk.Label(cf, text="Output: results/<material_folder>/Mode_<mode>_fd<value>_<material>.gif  +  .html",
                  foreground="#666", font=("", 8)).grid(
            row=2, column=0, columnspan=7, padx=10, pady=(0, 6), sticky="w")

        lf = ttk.LabelFrame(p, text="Log")
        lf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._alog = tk.Text(lf, height=12, state="disabled",
                              font=("Consolas", 9), bg="#1a1a2e", fg="#e0e0e0")
        sb = ttk.Scrollbar(lf, command=self._alog.yview)
        self._alog.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._alog.pack(fill="both", expand=True, padx=4, pady=4)

    # =======================================================================
    # TAB 5 — EXPORT
    # =======================================================================
    def _build_export(self):
        p = self.tab_exp

        bf = ttk.Frame(p)
        bf.pack(fill="x", padx=12, pady=14)
        ttk.Button(bf, text="Save All Results  →  TXT",
                   command=self._save_results).pack(side="left", padx=8, ipady=5)
        ttk.Button(bf, text="Open Results Folder",
                   command=self._open_results_folder).pack(side="left", padx=8, ipady=5)
        self._export_folder_var = tk.StringVar(value=f"Outputs:  {RESULTS_DIR}  (per-material subfolders)")
        ttk.Label(bf, textvariable=self._export_folder_var,
                  foreground="#666").pack(side="left", padx=12)

        lf = ttk.LabelFrame(p, text="Activity Log")
        lf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._elog = tk.Text(lf, height=24, state="disabled",
                              font=("Consolas", 9), bg="#1a1a2e", fg="#e0e0e0")
        sb = ttk.Scrollbar(lf, command=self._elog.yview)
        self._elog.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._elog.pack(fill="both", expand=True, padx=4, pady=4)

    # =======================================================================
    # PRESET / CLT LOGIC
    # =======================================================================
    def _load_preset(self, name):
        d = PRESETS[name]
        self._pv["e1"].set(str(d["E1"]))
        self._pv["e2"].set(str(d["E2"]))
        self._pv["g12"].set(str(d["G12"]))
        self._pv["nu12"].set(str(d["nu12"]))
        self._pv["rho"].set(str(d["rho"]))
        self._pv["thickness"].set(str(d["thickness"]))
        self._sv["mat_name"].set(name)
        self._compute_vel()

    def _clear_fields(self):
        for v in self._pv.values():
            v.set("")
        for v in self._vv.values():
            v.set("")
        self._sv["mat_name"].set("")

    def _compute_vel(self):
        try:
            cL, cS, cR = clt_velocities(
                float(self._pv["e1"].get()),
                float(self._pv["e2"].get()),
                float(self._pv["g12"].get()),
                float(self._pv["nu12"].get()),
                float(self._pv["rho"].get()),
            )
            self._vv["cL"].set(str(cL))
            self._vv["cS"].set(str(cS))
            self._vv["cR"].set(str(cR))
        except ValueError as e:
            messagebox.showerror("Input Error", f"Bad material property:\n{e}")

    # =======================================================================
    # SOLVE LOGIC
    # =======================================================================
    def _start_solve(self):
        try:
            thickness      = float(self._pv["thickness"].get())
            cL             = float(self._vv["cL"].get())
            cS             = float(self._vv["cS"].get())
            cR_str         = self._vv["cR"].get().strip()
            cR             = float(cR_str) if cR_str else None
            nmodes_sym     = int(self._sv["nmodes_sym"].get())
            nmodes_antisym = int(self._sv["nmodes_antisym"].get())
            fd_max         = float(self._sv["fd_max"].get())
            vp_max         = float(self._sv["vp_max"].get())
            fd_points      = int(self._sv["fd_points"].get())
            vp_step        = int(self._sv["vp_step"].get())
            mat_name       = self._sv["mat_name"].get().strip()
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter:\n{e}")
            return

        self._solve_btn.state(["disabled"])
        self._status.set("Solving…  (30–60 s depending on settings — please wait)")

        def _thread():
            try:
                lmb = Lamb(
                    thickness      = thickness,
                    nmodes_sym     = nmodes_sym,
                    nmodes_antisym = nmodes_antisym,
                    fd_max         = fd_max,
                    vp_max         = vp_max,
                    c_L            = cL,
                    c_S            = cS,
                    c_R            = cR,
                    fd_points      = fd_points,
                    vp_step        = vp_step,
                    material       = mat_name,
                )
                self.after(0, lambda: self._solve_done(lmb, nmodes_sym, nmodes_antisym, mat_name))
            except Exception as exc:
                self.after(0, lambda e=exc: self._solve_err(str(e)))

        threading.Thread(target=_thread, daemon=True).start()

    def _solve_done(self, lmb, ns, na, name):
        self.lamb = lmb
        self._solve_btn.state(["!disabled"])
        self._status.set(f"Done!  —  {name or 'custom material'}  |  {ns} sym + {na} antisym modes")
        modes = [f"A{i}" for i in range(na)] + [f"S{i}" for i in range(ns)]
        for cb in (self._ws_mode_cb, self._a_mode_cb, self._abq_mode_cb):
            cb["values"] = modes
        self._ws_mode.set("A0")
        self._a_mode.set("A0")
        self._abq_mode.set("A0")
        self._log(f"Solved: {name or 'custom'}  ({ns} sym + {na} antisym modes)")

    def _solve_err(self, msg):
        self._solve_btn.state(["!disabled"])
        self._status.set("Error — see dialog")
        messagebox.showerror("Solve Error", msg)

    # =======================================================================
    # DISPERSION PLOT HELPERS
    # =======================================================================
    def _req_lamb(self):
        if self.lamb is None:
            messagebox.showwarning("Not solved", "Please run SOLVE first (Setup tab).")
            return False
        return True

    def _close_fig(self, key):
        if key in self._figs:
            try:
                plt.close(self._figs.pop(key))
            except Exception:
                pass

    def _embed(self, frame, fig):
        for w in frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        tb = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _plot_phase(self):
        if not self._req_lamb(): return
        self._close_fig("disp")
        fig, _ = self.lamb.plot_phase_velocity(
            modes              = self._d_modes.get(),
            cutoff_frequencies = self._d_cutoff.get(),
            material_velocities= self._d_matvel.get(),
            save_img           = False,
            sym_style          = {"color": self._sym_col},
            antisym_style      = {"color": self._anti_col},
        )
        self._figs["disp"] = fig
        self._embed(self._df, fig)
        if self._d_save.get():
            out_dir = self.get_material_output_dir()
            safe = (self._sv["mat_name"].get().strip() or "custom").replace("/", "_").replace(" ", "_")
            thick = self._pv["thickness"].get().strip() or "0"
            path = os.path.join(out_dir, f"Phase_Velocity_{safe}_{thick}mm.png")
            fig.savefig(path, bbox_inches="tight")
            self._log(f"Phase velocity PNG saved → {out_dir}")

    def _plot_group(self):
        if not self._req_lamb(): return
        self._close_fig("disp")
        fig, _ = self.lamb.plot_group_velocity(
            modes              = self._d_modes.get(),
            cutoff_frequencies = self._d_cutoff.get(),
            save_img           = False,
            sym_style          = {"color": self._sym_col},
            antisym_style      = {"color": self._anti_col},
        )
        self._figs["disp"] = fig
        self._embed(self._df, fig)
        if self._d_save.get():
            out_dir = self.get_material_output_dir()
            safe = (self._sv["mat_name"].get().strip() or "custom").replace("/", "_").replace(" ", "_")
            thick = self._pv["thickness"].get().strip() or "0"
            path = os.path.join(out_dir, f"Group_Velocity_{safe}_{thick}mm.png")
            fig.savefig(path, bbox_inches="tight")
            self._log(f"Group velocity PNG saved → {out_dir}")

    def _plot_wnum(self):
        if not self._req_lamb(): return
        self._close_fig("disp")
        fig, _ = self.lamb.plot_wave_number(
            modes         = self._d_modes.get(),
            save_img      = False,
            sym_style     = {"color": self._sym_col},
            antisym_style = {"color": self._anti_col},
        )
        self._figs["disp"] = fig
        self._embed(self._df, fig)
        if self._d_save.get():
            out_dir = self.get_material_output_dir()
            safe = (self._sv["mat_name"].get().strip() or "custom").replace("/", "_").replace(" ", "_")
            thick = self._pv["thickness"].get().strip() or "0"
            path = os.path.join(out_dir, f"Wave_Number_{safe}_{thick}mm.png")
            fig.savefig(path, bbox_inches="tight")
            self._log(f"Wave number PNG saved → {out_dir}")

    # =======================================================================
    # WAVE STRUCTURE LOGIC
    # =======================================================================
    def _plot_wave_struct(self):
        if not self._req_lamb(): return
        try:
            mode  = self._ws_mode.get()
            nrows = int(self._ws_nrows.get())
            ncols = int(self._ws_ncols.get())
            n     = nrows * ncols
            fd_raw = self._ws_fd.get().strip()
            if fd_raw.lower() == "auto":
                fd_max = float(self._sv["fd_max"].get())
                fd_vals = np.linspace(fd_max / n, fd_max, n).tolist()
            else:
                fd_vals = [float(v.strip()) for v in fd_raw.split(",")]
                if len(fd_vals) != n:
                    messagebox.showerror(
                        "Input Error",
                        f"Grid is {nrows}×{ncols} = {n} subplots but {len(fd_vals)} fd values given.")
                    return
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        self._close_fig("wstr")
        fig, _ = self.lamb.plot_wave_structure(
            mode     = mode,
            nrows    = nrows,
            ncols    = ncols,
            fd       = fd_vals,
            save_img = False,
        )
        self._figs["wstr"] = fig
        self._embed(self._wsf, fig)
        if self._ws_save.get():
            out_dir = self.get_material_output_dir()
            safe = (self._sv["mat_name"].get().strip() or "custom").replace("/", "_").replace(" ", "_")
            path = os.path.join(out_dir, f"Wave_Structure_{mode}_{safe}.png")
            fig.savefig(path, bbox_inches="tight")
            self._log(f"Wave structure ({mode}) PNG saved → {out_dir}")

    # =======================================================================
    # ANIMATION LOGIC
    # =======================================================================
    def _gen_anim(self):
        if not self._req_lamb(): return
        try:
            mode = self._a_mode.get()
            fd   = float(self._a_fd.get())
            spd  = self._a_speed.get()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e)); return

        try:
            interp = (self.lamb.vp_antisym[mode] if mode[0] == "A"
                      else self.lamb.vp_sym[mode])
            fd_clamped = float(np.clip(fd, interp.x[0], interp.x[-1]))
            vp = float(interp(fd_clamped))
        except (KeyError, Exception) as e:
            messagebox.showerror("Error", f"Cannot query {mode} at fd={fd}:\n{e}"); return

        self._alog_w(f"Mode {mode}  |  fd = {fd} kHz·mm  |  vp = {vp:.0f} m/s  |  interval = {spd} ms")
        self._alog_w("Building 30 frames…")
        self._anim_btn.state(["disabled"])

        def _thread():
            try:
                x, y, times, fu, fw, maxd, wl = build_anim_data(self.lamb, mode, vp, fd)

                color = "#ff6b6b" if mode[0] == "A" else "#74b9ff"
                fig_a, ax_a = plt.subplots(figsize=(10, 4))
                fig_a.patch.set_facecolor("#0d1117")
                ax_a.set_facecolor("#0d1117")
                ax_a.tick_params(colors="white")
                for sp in ax_a.spines.values():
                    sp.set_edgecolor("#444444")
                ax_a.set_title(f"Mode {mode}  |  vp = {vp:.0f} m/s  |  fd = {fd:.0f} kHz·mm",
                               color="white", fontsize=10)
                ax_a.set_xlabel("x  (m)", color="white")
                ax_a.set_ylabel("Thickness", color="white")
                ax_a.axhline(0, color="#444444", lw=0.8, ls="--")
                h_m = self.lamb.h
                ax_a.axhline( h_m, color=color, lw=1.0, alpha=0.4)
                ax_a.axhline(-h_m, color=color, lw=1.0, alpha=0.4)

                q = ax_a.quiver(x, y, fu[0], fw[0],
                                scale=6*maxd, scale_units="inches",
                                color=color, alpha=0.85, width=0.003)
                plt.tight_layout()

                def _upd(i):
                    q.set_UVC(fu[i], fw[i])
                    return (q,)

                ani = anim_mod.FuncAnimation(fig_a, _upd, frames=len(times),
                                              interval=spd, blit=True, repeat=True)
                mat  = self._sv["mat_name"].get().strip() or "custom"
                safe = mat.replace("/", "_").replace(" ", "_")
                out_dir = self.get_material_output_dir()
                gif  = os.path.join(out_dir, f"Mode_{mode}_fd{int(fd)}_{safe}.gif")
                html = os.path.join(out_dir, f"Mode_{mode}_fd{int(fd)}_{safe}.html")

                ani.save(gif, writer=anim_mod.PillowWriter(fps=16),
                         dpi=90, savefig_kwargs={"facecolor": "#0d1117"})
                ani.save(html, writer=anim_mod.HTMLWriter(fps=16, embed_frames=True))
                plt.close(fig_a)
                self.after(0, lambda: self._anim_done(gif, html))
            except Exception as exc:
                self.after(0, lambda e=exc: self._anim_err(str(e)))

        threading.Thread(target=_thread, daemon=True).start()

    def _anim_done(self, gif, html):
        self._anim_btn.state(["!disabled"])
        self._alog_w(f"GIF  saved  ({os.path.getsize(gif) // 1024} KB)  →  {gif}")
        self._alog_w(f"HTML saved  ({os.path.getsize(html) // 1024} KB)  →  {html}")
        self._alog_w("Opening HTML in browser…")
        webbrowser.open(html)
        self._log(f"Animation: {os.path.basename(gif)}")

    def _anim_err(self, msg):
        self._anim_btn.state(["!disabled"])
        self._alog_w(f"ERROR: {msg}")
        messagebox.showerror("Animation Error", msg)

    def _alog_w(self, msg):
        self._alog.config(state="normal")
        self._alog.insert("end", msg + "\n")
        self._alog.see("end")
        self._alog.config(state="disabled")

    # =======================================================================
    # EXPORT LOGIC
    # =======================================================================
    def _save_results(self):
        if not self._req_lamb(): return
        try:
            out_dir = self.get_material_output_dir()
            old_cwd = os.getcwd()
            try:
                os.chdir(out_dir)
                os.makedirs("results", exist_ok=True)
                self.lamb.save_results()
                for f in os.listdir("results"):
                    shutil.move(os.path.join("results", f), f)
                if os.path.isdir("results"):
                    os.rmdir("results")
            finally:
                os.chdir(old_cwd)
            self._log(f"Results TXT saved → {out_dir}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _open_results_folder(self):
        """Open the material-specific output folder if available, else the main results folder."""
        try:
            out_dir = self.get_material_output_dir()
            if os.path.isdir(out_dir):
                os.startfile(out_dir)
            else:
                os.startfile(RESULTS_DIR)
        except Exception:
            os.startfile(RESULTS_DIR)

    def _log(self, msg):
        self._elog.config(state="normal")
        self._elog.insert("end", msg + "\n")
        self._elog.see("end")
        self._elog.config(state="disabled")

    # =======================================================================
    # TAB 5 — ABAQUS PARAMETERS
    # =======================================================================
    def _build_abaqus(self):
        p = self.tab_abq

        # ---- Top: inputs + results side by side ----
        main = ttk.Frame(p)
        main.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        left  = ttk.LabelFrame(main, text="Excitation & Mesh Inputs")
        right = ttk.LabelFrame(main, text="Computed Parameters  (copy into Abaqus)")
        left.pack(side="left", fill="y", padx=(0, 6))
        right.pack(side="left", fill="both", expand=True)

        input_defs = [
            ("Excitation frequency (kHz)  (?)", "freq",  "50",  "freq"),
            ("Elements per wavelength  (?)",    "epw",   "10",  "epw"),
            ("Elements through thickness  (?)", "eth",   "2",   "eth"),
            ("CFL safety factor  (?)",          "cfl",   "0.9", "cfl"),
            ("Tone burst cycles  (?)",          "cyc",   "3.5", "cyc"),
            ("Amplitude  (mm)  (?)",            "amp",   "0.05","amp"),
            ("Total simulation time (µs)  (?)", "tsim",  "300", "tsim"),
            ("Plate  L  (mm)  (?)",             "pL",    "300", "pL"),
            ("Plate  W  (mm)  (?)",             "pW",    "300", "pW"),
        ]
        self._abq = {}
        for r, (lbl, key, dflt, tip_key) in enumerate(input_defs):
            lw = ttk.Label(left, text=lbl, foreground="#1a73e8", cursor="question_arrow")
            lw.grid(row=r, column=0, sticky="w", padx=10, pady=5)
            add_tip(lw, tip_key)
            v = tk.StringVar(value=dflt)
            ttk.Entry(left, textvariable=v, width=12).grid(row=r, column=1, padx=10, pady=5)
            self._abq[key] = v

        ttk.Label(left, text="Target mode:").grid(
            row=len(input_defs), column=0, sticky="w", padx=10, pady=5)
        self._abq_mode = tk.StringVar(value="A0")
        self._abq_mode_cb = ttk.Combobox(left, textvariable=self._abq_mode,
                                          values=["A0"], width=10, state="readonly")
        self._abq_mode_cb.grid(row=len(input_defs), column=1, padx=10, pady=5)

        ttk.Button(left, text="Compute Abaqus Parameters",
                   command=self._compute_abaqus
                   ).grid(row=len(input_defs)+1, column=0, columnspan=2,
                          padx=10, pady=(14, 6), sticky="ew")

        # Results text box
        self._abq_txt = tk.Text(right, state="disabled", font=("Consolas", 9),
                                 bg="#0d1117", fg="#e0e0e0", height=22)
        sb = ttk.Scrollbar(right, command=self._abq_txt.yview)
        self._abq_txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._abq_txt.pack(fill="both", expand=True, padx=4, pady=4)
        self._abq_txt.config(state="normal")
        self._abq_txt.insert("end", "  Solve (Setup tab) → fill inputs → click Compute\n")
        self._abq_txt.config(state="disabled")

        # ---- Bottom: export buttons ----
        ef = ttk.LabelFrame(p, text="Export")
        ef.pack(fill="x", padx=10, pady=(2, 10))

        # Row 1 — individual exports
        row1 = ttk.Frame(ef)
        row1.pack(fill="x", padx=4, pady=(6, 2))
        for lbl, cmd in [
            ("Dispersion + Op.Point  (PNG)", self._export_disp_fig),
            ("Tone Burst  (PNG)",            self._export_toneburst_png),
            ("Tone Burst  (CSV)",            self._export_toneburst_csv),
            ("Tone Burst FFT  (PNG)",        self._export_toneburst_fft),
            ("Summary Card  (PNG)",          self._export_summary_card),
            ("Parameters  (TXT)",            self._export_params_txt),
        ]:
            ttk.Button(row1, text=lbl, command=cmd).pack(side="left", padx=5, ipady=4)

        # Row 2 — combined
        row2 = ttk.Frame(ef)
        row2.pack(fill="x", padx=4, pady=(2, 8))
        ttk.Button(row2, text="  Export All  (6 files)  ",
                   command=self._export_all_abaqus).pack(side="left", padx=5, ipady=4)

    # =======================================================================
    # ABAQUS COMPUTATION
    # =======================================================================
    def _compute_abaqus(self):
        if not self._req_lamb(): return
        try:
            freq_khz = float(self._abq["freq"].get())
            epw      = float(self._abq["epw"].get())
            eth      = int(self._abq["eth"].get())
            cfl      = float(self._abq["cfl"].get())
            cycles   = float(self._abq["cyc"].get())
            amp_mm   = float(self._abq["amp"].get())
            t_sim    = float(self._abq["tsim"].get())
            plate_L  = float(self._abq["pL"].get())
            plate_W  = float(self._abq["pW"].get())
            mode     = self._abq_mode.get()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e)); return

        try:
            thickness_mm = float(self._pv["thickness"].get())
            c_L          = float(self._vv["cL"].get())
        except ValueError:
            messagebox.showerror("Error", "Thickness / c_L missing in Setup tab."); return

        mat_name = self._sv["mat_name"].get().strip() or "Custom"
        freq_hz  = freq_khz * 1e3
        fd       = freq_khz * thickness_mm

        # Velocities at operating fd
        try:
            if mode[0] == "A":
                interp_vp = self.lamb.vp_antisym[mode]
                interp_vg = self.lamb.vg_antisym[mode]
            else:
                interp_vp = self.lamb.vp_sym[mode]
                interp_vg = self.lamb.vg_sym[mode]
            fd_clamped = float(np.clip(fd, interp_vp.x[0], interp_vp.x[-1]))
            vp = float(interp_vp(fd_clamped))
            vg = float(interp_vg(fd_clamped))
        except (KeyError, Exception) as e:
            messagebox.showerror("Error", f"Cannot query {mode} at fd={fd}:\n{e}"); return

        # S0 reference (best-effort)
        try:
            vp_s0 = float(self.lamb.vp_sym["S0"](fd))
            vg_s0 = float(self.lamb.vg_sym["S0"](fd))
            wl_s0 = (vp_s0 / freq_hz) * 1e3
        except Exception:
            vp_s0 = vg_s0 = wl_s0 = None

        wl_mm       = (vp / freq_hz) * 1e3
        elem_mm     = wl_mm / epw
        Le_m        = elem_mm / 1000.0
        dt          = cfl * Le_m / (c_L * (3 ** 0.5))
        t1_us       = cycles / freq_hz * 1e6
        n_x         = int(np.ceil(plate_L / elem_mm))
        n_y         = int(np.ceil(plate_W / elem_mm))
        n_total     = n_x * n_y * eth
        n_steps     = int(t_sim * 1e-6 / dt)

        self._abq_params = dict(
            mat_name=mat_name, freq_khz=freq_khz, mode=mode, fd=fd,
            vp=vp, vg=vg, wl_mm=wl_mm,
            vp_s0=vp_s0, vg_s0=vg_s0, wl_s0=wl_s0,
            c_L=c_L, elem_mm=elem_mm, eth=eth, epw=epw,
            dt=dt, cfl=cfl, cycles=cycles, amp_mm=amp_mm,
            t1_us=t1_us, t_sim=t_sim,
            plate_L=plate_L, plate_W=plate_W, thickness_mm=thickness_mm,
            n_x=n_x, n_y=n_y, n_z=eth, n_total=n_total, n_steps=n_steps,
        )

        D = "=" * 56
        d = "-" * 52
        lines = [
            D,
            f"  ABAQUS PARAMETERS  —  {mat_name}  @  {freq_khz:.0f} kHz",
            D, "",
            f"  OPERATING POINT   fd = {fd:.2f} kHz·mm",
            f"  {d}",
            f"  Target mode         {mode}",
            f"  Phase velocity  vp = {vp:>9.1f}  m/s",
            f"  Group velocity  vg = {vg:>9.1f}  m/s",
            f"  Wavelength       λ = {wl_mm:>9.3f}  mm",
        ]
        if vp_s0:
            lines += [
                "",
                f"  S0 (for reference)  vp = {vp_s0:>9.1f}  m/s",
                f"                      vg = {vg_s0:>9.1f}  m/s",
                f"                       λ = {wl_s0:>9.3f}  mm",
            ]
        lines += [
            "", f"  MESH  (element type C3D8R)",
            f"  {d}",
            f"  In-plane size  Le = {elem_mm:>9.4f}  mm   (λ / {epw:.0f})",
            f"  Through thickness  {eth} elements  ({thickness_mm/eth:.4f} mm/elem)",
            f"  Elements X         {n_x}   ({plate_L:.0f} mm / {elem_mm:.4f} mm/elem)",
            f"  Elements Y         {n_y}   ({plate_W:.0f} mm / {elem_mm:.4f} mm/elem)",
            f"  Elements Z         {eth}",
            f"  Total elements ≈   {n_total:,}",
            "", f"  TIME STEPPING",
            f"  {d}",
            f"  dt (CFL×{cfl})   =  {dt:.4e}  s",
            f"  Total time     =  {t_sim:.0f}  µs",
            f"  Steps (est.)   ≈  {n_steps:,}",
            "", f"  EXCITATION  (Hanning-windowed tone burst, mode {mode})",
            f"  {d}",
            f"  Frequency   fc = {freq_khz:.0f}  kHz",
            f"  Cycles         = {cycles}",
            f"  Duration    t1 = {t1_us:.3f}  µs",
            f"  Amplitude    A = {amp_mm}  mm   (applied as U2)",
            "",
            f"  U2(t) = -{amp_mm}/2 * sin(2*pi*{freq_khz:.0f}e3*t)",
            f"          * (1 - cos(2*pi*{freq_khz:.0f}e3*t / {cycles}))",
            f"  for  0 <= t <= {t1_us:.3f} µs,  else  U2 = 0",
            "", D,
        ]

        self._abq_txt.config(state="normal")
        self._abq_txt.delete("1.0", "end")
        self._abq_txt.insert("end", "\n".join(lines))
        self._abq_txt.config(state="disabled")
        self._log(f"Abaqus params computed: {mat_name} @ {freq_khz:.0f} kHz  |  Le={elem_mm:.3f} mm  dt={dt:.2e} s")

    # =======================================================================
    # ABAQUS EXPORT METHODS
    # =======================================================================
    def _req_abq(self):
        if self._abq_params is None:
            messagebox.showwarning("No data", "Click 'Compute Abaqus Parameters' first.")
            return False
        return True

    def _safe_name(self):
        d = self._abq_params
        return f"{d['mat_name'].replace('/','_')}_{d['freq_khz']:.0f}kHz"

    # --- Figure 1: Dispersion curves with operating point ---
    def _export_disp_fig(self):
        if not self._req_abq(): return
        d   = self._abq_params
        fd_max = float(self._sv["fd_max"].get())
        fd_arr = np.linspace(0.01, fd_max, 600)

        BG = "#0d1117"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.patch.set_facecolor(BG)
        for ax in (ax1, ax2):
            ax.set_facecolor(BG)
            ax.tick_params(colors="white");  ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white");  ax.title.set_color("white")
            for sp in ax.spines.values():  sp.set_edgecolor("#333")

        # Plot all modes
        anti_modes = list(self.lamb.vp_antisym.keys())
        sym_modes  = list(self.lamb.vp_sym.keys())
        n_a = len(anti_modes);  n_s = len(sym_modes)
        anti_cols = plt.cm.Reds(np.linspace(0.45, 0.95, max(n_a, 1)))
        sym_cols  = plt.cm.Blues(np.linspace(0.45, 0.95, max(n_s, 1)))

        for i, m in enumerate(anti_modes):
            vp_i = self.lamb.vp_antisym[m]
            vg_i = self.lamb.vg_antisym[m]
            fd_m = fd_arr[(fd_arr >= vp_i.x[0]) & (fd_arr <= vp_i.x[-1])]
            if len(fd_m) == 0:
                continue
            lw = 2.0 if m == d["mode"] else 1.0
            ax1.plot(fd_m, vp_i(fd_m), color=anti_cols[i], lw=lw, label=m)
            ax2.plot(fd_m, vg_i(fd_m), color=anti_cols[i], lw=lw, label=m)

        for i, m in enumerate(sym_modes):
            vp_i = self.lamb.vp_sym[m]
            vg_i = self.lamb.vg_sym[m]
            fd_m = fd_arr[(fd_arr >= vp_i.x[0]) & (fd_arr <= vp_i.x[-1])]
            if len(fd_m) == 0:
                continue
            lw = 2.0 if m == d["mode"] else 1.0
            ax1.plot(fd_m, vp_i(fd_m), color=sym_cols[i], lw=lw, label=m)
            ax2.plot(fd_m, vg_i(fd_m), color=sym_cols[i], lw=lw, label=m)

        # Operating point
        for ax, val, lbl in [(ax1, d["vp"], f"vp={d['vp']:.0f} m/s"),
                              (ax2, d["vg"], f"vg={d['vg']:.0f} m/s")]:
            ax.axvline(d["fd"], color="#ffdd57", lw=1.2, ls="--", alpha=0.7)
            ax.plot(d["fd"], val, marker="*", ms=14, ls="none", color="#ffdd57", zorder=5)
            ax.annotate(
                f"  {d['mode']}  {lbl}\n  λ={d['wl_mm']:.1f} mm",
                xy=(d["fd"], val), xytext=(d["fd"] + fd_max*0.03, val),
                color="#ffdd57", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#ffdd57", lw=0.8),
            )

        ax1.set_ylabel("Phase Velocity  (m/s)", color="white")
        ax2.set_ylabel("Group Velocity  (m/s)", color="white")
        ax2.set_xlabel("Frequency × Thickness  (kHz·mm)", color="white")
        ax1.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
                   loc="upper left", ncol=2)
        fig.suptitle(
            f"Lamb Wave Dispersion  —  {d['mat_name']}  |  "
            f"h = {d['thickness_mm']:.3f} mm  |  Operating point: {d['freq_khz']:.0f} kHz",
            color="white", fontsize=11, y=0.99)
        plt.tight_layout()

        path = os.path.join(self.get_material_output_dir(), f"dispersion_curves_{self._safe_name()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        self._log(f"Dispersion figure saved → {path}")
        messagebox.showinfo("Saved", f"Dispersion figure saved:\n{path}")

    # --- Figure 2: Tone burst signal (PNG) + Abaqus CSV + FFT spectrum PNG ---
    # ------------------------------------------------------------------
    # Shared signal builder — called by all three tone-burst exports
    # ------------------------------------------------------------------
    def _toneburst_signal(self):
        """Return signal arrays for the current Abaqus params dict."""
        d       = self._abq_params
        freq_hz = d["freq_khz"] * 1e3
        t1      = d["t1_us"] * 1e-6
        dt_abq  = d["dt"]

        # High-res array for smooth time-domain plots (2 000 points)
        t_plot = np.linspace(0, t1 * 1.1, 2000)
        y_plot = np.where(
            t_plot <= t1,
            -d["amp_mm"] / 2 * np.sin(2 * np.pi * freq_hz * t_plot)
            * (1 - np.cos(2 * np.pi * freq_hz * t_plot / d["cycles"])),
            0.0,
        )

        # Abaqus-rate array (uniform dt, used for CSV and FFT)
        t_csv = np.arange(0.0, t1 + dt_abq, dt_abq)
        y_csv = np.where(
            t_csv <= t1,
            -d["amp_mm"] / 2 * np.sin(2 * np.pi * freq_hz * t_csv)
            * (1 - np.cos(2 * np.pi * freq_hz * t_csv / d["cycles"])),
            0.0,
        )
        return t_plot, y_plot, t_csv, y_csv, freq_hz, t1

    # --- Figure 2a: Time-domain waveform PNG ---
    def _export_toneburst_png(self):
        if not self._req_abq(): return
        d    = self._abq_params
        BG   = "#0d1117"
        t_plot, y_plot, _tc, _yc, _fhz, t1 = self._toneburst_signal()

        fig, ax = plt.subplots(figsize=(11, 4))
        fig.patch.set_facecolor(BG);  ax.set_facecolor(BG)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():  sp.set_edgecolor("#333")
        ax.plot(t_plot * 1e6, y_plot, color="#74b9ff", lw=1.8)
        ax.axvline(d["t1_us"], color="#ffdd57", lw=1.0, ls="--", alpha=0.7,
                   label=f"t1 = {d['t1_us']:.2f} µs  (end of burst)")
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_xlabel("Time  (µs)", color="white")
        ax.set_ylabel("U2 Displacement  (mm)", color="white")
        ax.set_title(
            f"Hanning-Windowed Tone Burst  —  {d['cycles']} cycles @ {d['freq_khz']:.0f} kHz  "
            f"|  A = {d['amp_mm']} mm  |  {d['mat_name']}",
            color="white", fontsize=10)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax.set_xlim(0, t_plot[-1] * 1e6)
        ax.xaxis.label.set_color("white");  ax.yaxis.label.set_color("white")
        plt.tight_layout()

        path = os.path.join(self.get_material_output_dir(), f"tone_burst_{self._safe_name()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        self._log(f"Tone burst PNG  saved → {path}")
        messagebox.showinfo("Saved", f"Tone burst waveform saved:\n{path}")

    # --- Figure 2b: Abaqus amplitude CSV ---
    def _export_toneburst_csv(self):
        if not self._req_abq(): return
        d = self._abq_params
        _tp, _yp, t_csv, y_csv, _fhz, _t1 = self._toneburst_signal()

        path  = os.path.join(self.get_material_output_dir(), f"tone_burst_{self._safe_name()}.csv")
        np.savetxt(path,
                   np.column_stack([t_csv, y_csv]),
                   delimiter=",",
                   fmt="%.8e")
        n_pts = len(t_csv)
        self._log(f"Tone burst CSV  saved → {path}  ({n_pts} rows)")
        messagebox.showinfo(
            "Saved",
            f"Abaqus amplitude CSV saved:\n{path}\n\n"
            f"{n_pts} rows  |  dt = {d['dt']:.3e} s\n\n"
            "Import in Abaqus:  Amplitude → Tabular → read from this file."
        )

    # --- Figure 2c: FFT spectrum PNG ---
    def _export_toneburst_fft(self):
        if not self._req_abq(): return
        d  = self._abq_params
        BG = "#0d1117"
        _tp, _yp, t_csv, y_csv, _fhz, _t1 = self._toneburst_signal()
        dt_abq = d["dt"]

        spectrum  = np.abs(np.fft.rfft(y_csv))
        freqs     = np.fft.rfftfreq(len(y_csv), d=dt_abq)
        spec_norm = spectrum / spectrum.max()

        fc_fft_hz  = freqs[np.argmax(spectrum)]
        fc_fft_khz = fc_fft_hz / 1e3

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(BG);  ax.set_facecolor(BG)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():  sp.set_edgecolor("#333")

        f_khz   = freqs / 1e3
        f_limit = d["freq_khz"] * 3.0
        mask    = f_khz <= f_limit
        ax.fill_between(f_khz[mask], spec_norm[mask], color="#74b9ff", alpha=0.4)
        ax.plot(f_khz[mask], spec_norm[mask], color="#74b9ff", lw=1.5)
        ax.axvline(fc_fft_khz, color="#ffdd57", lw=1.2, ls="--",
                   label=f"Peak at {fc_fft_khz:.1f} kHz")
        ax.axvline(d["freq_khz"], color="#ff6b6b", lw=0.8, ls=":",
                   label=f"fc = {d['freq_khz']:.0f} kHz (nominal)")
        ax.set_xlabel("Frequency  (kHz)", color="white")
        ax.set_ylabel("Normalised Magnitude", color="white")
        ax.set_title(
            f"FFT Spectrum of Tone Burst  —  {d['cycles']} cycles @ {d['freq_khz']:.0f} kHz  "
            f"|  {d['mat_name']}",
            color="white", fontsize=10)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax.set_xlim(0, f_limit)
        ax.xaxis.label.set_color("white");  ax.yaxis.label.set_color("white")
        plt.tight_layout()

        path = os.path.join(self.get_material_output_dir(), f"tone_burst_FFT_{self._safe_name()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        self._log(f"Tone burst FFT  saved → {path}")
        messagebox.showinfo("Saved", f"FFT spectrum saved:\n{path}\n\nFFT peak: {fc_fft_khz:.1f} kHz")

    # --- Combined: all three tone-burst files (used by Export All) ---
    def _export_toneburst(self):
        if not self._req_abq(): return
        d = self._abq_params
        _tp, _yp, t_csv, y_csv, _fhz, _t1 = self._toneburst_signal()

        # Reuse individual methods (they each show their own messagebox when called
        # standalone, but here we suppress the individual dialogs and show one summary)
        safe = self._safe_name()
        BG   = "#0d1117"

        # PNG
        t_plot, y_plot, t_csv, y_csv, freq_hz, t1 = self._toneburst_signal()
        dt_abq = d["dt"]
        fig1, ax1 = plt.subplots(figsize=(11, 4))
        fig1.patch.set_facecolor(BG);  ax1.set_facecolor(BG)
        ax1.tick_params(colors="white")
        for sp in ax1.spines.values():  sp.set_edgecolor("#333")
        ax1.plot(t_plot * 1e6, y_plot, color="#74b9ff", lw=1.8)
        ax1.axvline(d["t1_us"], color="#ffdd57", lw=1.0, ls="--", alpha=0.7,
                    label=f"t1 = {d['t1_us']:.2f} µs  (end of burst)")
        ax1.axhline(0, color="#444", lw=0.6)
        ax1.set_xlabel("Time  (µs)", color="white");  ax1.set_ylabel("U2 Displacement  (mm)", color="white")
        ax1.set_title(
            f"Hanning-Windowed Tone Burst  —  {d['cycles']} cycles @ {d['freq_khz']:.0f} kHz  "
            f"|  A = {d['amp_mm']} mm  |  {d['mat_name']}", color="white", fontsize=10)
        ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax1.set_xlim(0, t_plot[-1] * 1e6)
        ax1.xaxis.label.set_color("white");  ax1.yaxis.label.set_color("white")
        plt.tight_layout()
        out_dir = self.get_material_output_dir()
        png_path = os.path.join(out_dir, f"tone_burst_{safe}.png")
        fig1.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=BG);  plt.close(fig1)
        self._log(f"Tone burst PNG  saved → {png_path}")

        # CSV
        csv_path = os.path.join(out_dir, f"tone_burst_{safe}.csv")
        np.savetxt(csv_path, np.column_stack([t_csv, y_csv]), delimiter=",", fmt="%.8e")
        n_pts = len(t_csv)
        self._log(f"Tone burst CSV  saved → {csv_path}  ({n_pts} rows)")

        # FFT
        spectrum  = np.abs(np.fft.rfft(y_csv))
        freqs     = np.fft.rfftfreq(len(y_csv), d=dt_abq)
        spec_norm = spectrum / spectrum.max()
        fc_fft_khz = freqs[np.argmax(spectrum)] / 1e3
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor(BG);  ax2.set_facecolor(BG)
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values():  sp.set_edgecolor("#333")
        f_khz = freqs / 1e3;  f_limit = d["freq_khz"] * 3.0;  mask = f_khz <= f_limit
        ax2.fill_between(f_khz[mask], spec_norm[mask], color="#74b9ff", alpha=0.4)
        ax2.plot(f_khz[mask], spec_norm[mask], color="#74b9ff", lw=1.5)
        ax2.axvline(fc_fft_khz, color="#ffdd57", lw=1.2, ls="--", label=f"Peak at {fc_fft_khz:.1f} kHz")
        ax2.axvline(d["freq_khz"], color="#ff6b6b", lw=0.8, ls=":", label=f"fc = {d['freq_khz']:.0f} kHz (nominal)")
        ax2.set_xlabel("Frequency  (kHz)", color="white");  ax2.set_ylabel("Normalised Magnitude", color="white")
        ax2.set_title(
            f"FFT Spectrum of Tone Burst  —  {d['cycles']} cycles @ {d['freq_khz']:.0f} kHz  "
            f"|  {d['mat_name']}", color="white", fontsize=10)
        ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        ax2.set_xlim(0, f_limit);  ax2.xaxis.label.set_color("white");  ax2.yaxis.label.set_color("white")
        plt.tight_layout()
        fft_path = os.path.join(out_dir, f"tone_burst_FFT_{safe}.png")
        fig2.savefig(fft_path, dpi=150, bbox_inches="tight", facecolor=BG);  plt.close(fig2)
        self._log(f"Tone burst FFT  saved → {fft_path}")

        # Summary stats
        starts_zero = abs(y_csv[0])  < 1e-12
        ends_zero   = abs(y_csv[-1]) < 1e-9
        self._log("─" * 52)
        self._log(f"  Total duration      : {d['t1_us']:.2f} µs")
        self._log(f"  CSV data points     : {n_pts}")
        self._log(f"  FFT centre freq     : {fc_fft_khz:.2f} kHz  (nominal {d['freq_khz']:.0f} kHz)")
        self._log(f"  Starts at zero      : {'YES' if starts_zero else 'NO  (check equation)'}")
        self._log(f"  Ends at zero        : {'YES' if ends_zero   else 'NO  (check equation)'}")
        self._log("─" * 52)

        messagebox.showinfo(
            "Tone Burst Exported",
            f"3 files saved to material folder:\n\n"
            f"  {os.path.basename(png_path)}\n"
            f"  {os.path.basename(csv_path)}\n"
            f"  {os.path.basename(fft_path)}\n\n"
            f"CSV: {n_pts} rows  |  FFT peak: {fc_fft_khz:.1f} kHz"
        )

    # --- Figure 3: Summary card ---
    def _export_summary_card(self):
        if not self._req_abq(): return
        d   = self._abq_params
        BG  = "#0d1117"
        FG  = "#e0e0e0"
        YEL = "#ffdd57"
        GRN = "#55efc4"
        BLU = "#74b9ff"

        fig = plt.figure(figsize=(14, 9))
        fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(BG);  ax.axis("off")

        def txt(x, y, s, **kw):
            ax.text(x, y, s, transform=ax.transAxes,
                    verticalalignment="top", **kw)

        txt(0.5, 0.97,
            f"ABAQUS SIMULATION PARAMETERS",
            color=YEL, fontsize=16, fontweight="bold", ha="center",
            fontfamily="monospace")
        txt(0.5, 0.92,
            f"{d['mat_name']}  |  {d['freq_khz']:.0f} kHz  |  "
            f"h = {d['thickness_mm']:.3f} mm  |  Mode {d['mode']}",
            color=FG, fontsize=11, ha="center", fontfamily="monospace")

        ax.plot([0, 1], [0.89, 0.89], color="#333", lw=1, transform=ax.transAxes)

        # Column 1 — Wave properties
        col1_x, col2_x, col3_x = 0.03, 0.37, 0.70
        y0 = 0.86
        dy = 0.055

        def section(x, y, title, rows, color):
            txt(x, y, title, color=color, fontsize=10, fontweight="bold",
                fontfamily="monospace")
            for i, (k, v) in enumerate(rows):
                txt(x+0.01, y - (i+1)*dy, k, color="#aaa", fontsize=9,
                    fontfamily="monospace")
                txt(x+0.20, y - (i+1)*dy, v, color=FG, fontsize=9,
                    fontfamily="monospace")

        section(col1_x, y0, "WAVE PROPERTIES", [
            ("Phase vel. vp",  f"{d['vp']:.1f} m/s"),
            ("Group vel. vg",  f"{d['vg']:.1f} m/s"),
            ("Wavelength  λ",  f"{d['wl_mm']:.3f} mm"),
            ("fd",             f"{d['fd']:.2f} kHz·mm"),
            ("S0 vp (ref.)",   f"{d['vp_s0']:.0f} m/s" if d['vp_s0'] else "—"),
            ("S0 vg (ref.)",   f"{d['vg_s0']:.0f} m/s" if d['vg_s0'] else "—"),
        ], BLU)

        section(col2_x, y0, "MESH  (C3D8R)", [
            ("In-plane Le",    f"{d['elem_mm']:.4f} mm"),
            ("λ / Le",         f"{d['epw']:.0f} elem/λ"),
            ("Thru-thickness", f"{d['eth']} elements"),
            ("Elements X",     f"{d['n_x']}"),
            ("Elements Y",     f"{d['n_y']}"),
            ("Total (est.)",   f"{d['n_total']:,}"),
        ], GRN)

        section(col3_x, y0, "TIME & EXCITATION", [
            ("dt (CFL)",       f"{d['dt']:.3e} s"),
            ("Safety factor",  f"{d['cfl']}"),
            ("Sim. time",      f"{d['t_sim']:.0f} µs"),
            ("Steps (est.)",   f"{d['n_steps']:,}"),
            ("Cycles",         f"{d['cycles']}"),
            ("Duration t1",    f"{d['t1_us']:.3f} µs"),
        ], "#fd79a8")

        # Tone burst equation box
        eq_y = y0 - 7.5*dy
        ax.plot([0, 1], [eq_y + 0.02, eq_y + 0.02], color="#333", lw=0.8, transform=ax.transAxes)
        txt(col1_x, eq_y, "EXCITATION EQUATION  (apply as U2 displacement on actuator nodes):",
            color=YEL, fontsize=9, fontweight="bold", fontfamily="monospace")
        eq1 = (f"U2(t) = -{d['amp_mm']}/2 × sin(2π × {d['freq_khz']:.0f}e3 × t)"
               f"  ×  (1 - cos(2π × {d['freq_khz']:.0f}e3 × t / {d['cycles']}))"
               f"       for  0 ≤ t ≤ {d['t1_us']:.3f} µs")
        txt(col1_x, eq_y - dy, eq1, color=FG, fontsize=9, fontfamily="monospace")
        txt(col1_x, eq_y - 2*dy,
            f"U2(t) = 0                                                      for  t > {d['t1_us']:.3f} µs",
            color=FG, fontsize=9, fontfamily="monospace")

        ax.plot([0, 1], [0.04, 0.04], color="#333", lw=0.8, transform=ax.transAxes)
        txt(0.5, 0.03,
            "Generated by Lamb Wave Dispersion — CFRP Toolkit",
            color="#555", fontsize=8, ha="center", fontfamily="monospace")

        path = os.path.join(self.get_material_output_dir(), f"abaqus_summary_{self._safe_name()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        self._log(f"Summary card saved → {path}")
        messagebox.showinfo("Saved", f"Summary card saved:\n{path}")

    # --- TXT export ---
    def _export_params_txt(self):
        if not self._req_abq(): return
        d    = self._abq_params
        path = os.path.join(self.get_material_output_dir(), f"abaqus_params_{self._safe_name()}.txt")
        lines = [
            "ABAQUS SIMULATION PARAMETERS",
            "=" * 60,
            f"Material              : {d['mat_name']}",
            f"Frequency             : {d['freq_khz']} kHz",
            f"Plate thickness       : {d['thickness_mm']} mm",
            f"fd (operating point)  : {d['fd']:.4f} kHz·mm",
            "",
            "WAVE PROPERTIES",
            "-" * 40,
            f"Target mode           : {d['mode']}",
            f"Phase velocity  vp    : {d['vp']:.2f} m/s",
            f"Group velocity  vg    : {d['vg']:.2f} m/s",
            f"Wavelength      lam   : {d['wl_mm']:.4f} mm",
        ]
        if d["vp_s0"]:
            lines += [
                f"S0 phase velocity     : {d['vp_s0']:.2f} m/s",
                f"S0 group velocity     : {d['vg_s0']:.2f} m/s",
                f"S0 wavelength         : {d['wl_s0']:.4f} mm",
            ]
        lines += [
            "",
            "MESH PARAMETERS",
            "-" * 40,
            f"Element type          : C3D8R",
            f"In-plane size  Le     : {d['elem_mm']:.6f} mm  (lam / {d['epw']:.0f})",
            f"Through thickness     : {d['eth']} elements",
            f"Elements X            : {d['n_x']}",
            f"Elements Y            : {d['n_y']}",
            f"Elements Z            : {d['n_z']}",
            f"Total elements (est.) : {d['n_total']}",
            "",
            "TIME STEPPING",
            "-" * 40,
            f"CFL safety factor     : {d['cfl']}",
            f"Time step dt          : {d['dt']:.6e} s",
            f"Total simulation time : {d['t_sim']} us",
            f"Steps (estimate)      : {d['n_steps']}",
            "",
            "EXCITATION",
            "-" * 40,
            f"Signal type           : Hanning-windowed tone burst",
            f"Cycles                : {d['cycles']}",
            f"Duration t1           : {d['t1_us']:.4f} us",
            f"Amplitude A           : {d['amp_mm']} mm  (U2 displacement)",
            "",
            "Equation (0 <= t <= t1):",
            f"  U2(t) = -{d['amp_mm']}/2 * sin(2*pi*{d['freq_khz']}e3*t)",
            f"          * (1 - cos(2*pi*{d['freq_khz']}e3*t / {d['cycles']}))",
            "  U2(t) = 0  for t > t1",
            "",
            "=" * 60,
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self._log(f"Parameters TXT saved → {path}")
        messagebox.showinfo("Saved", f"Parameters TXT saved:\n{path}")

    # --- Export all ---
    def _export_all_abaqus(self):
        if not self._req_abq(): return
        self._export_disp_fig()
        self._export_toneburst()
        self._export_summary_card()
        self._export_params_txt()
        messagebox.showinfo("Export All Done",
                            f"6 files saved to:\n{self.get_material_output_dir()}")

    # =======================================================================
    # TAB — HELP / GUIDE
    # =======================================================================
    def _build_help(self):
        p = self.tab_help

        hdr = ttk.Frame(p)
        hdr.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(
            hdr,
            text="Parameter Guide  —  hover over any blue (?) label for a quick tooltip, "
                 "or read the full reference below.",
            foreground="#555", font=("Segoe UI", 9, "italic"),
        ).pack(side="left")

        txt_frame = ttk.Frame(p)
        txt_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        sb  = ttk.Scrollbar(txt_frame)
        sb.pack(side="right", fill="y")

        t = tk.Text(
            txt_frame, font=("Consolas", 9), bg="#0d1117", fg="#e0e0e0",
            wrap="word", state="normal", yscrollcommand=sb.set,
            padx=12, pady=8,
        )
        t.pack(fill="both", expand=True)
        sb.config(command=t.yview)

        # ---- tag styles ----
        t.tag_config("h1",  foreground="#ffdd57", font=("Consolas", 11, "bold"))
        t.tag_config("h2",  foreground="#74b9ff", font=("Consolas", 10, "bold"))
        t.tag_config("key", foreground="#55efc4", font=("Consolas", 9,  "bold"))
        t.tag_config("tip", foreground="#dfe6e9", font=("Consolas", 9))
        t.tag_config("dim", foreground="#888",    font=("Consolas", 8,  "italic"))
        t.tag_config("sep", foreground="#333333")

        def h1(text):
            t.insert("end", f"\n{text}\n", "h1")
        def h2(text):
            t.insert("end", f"\n  {text}\n", "h2")
        def row(label, body):
            t.insert("end", f"    {label:<32}", "key")
            t.insert("end", f"{body}\n", "tip")
        def note(text):
            t.insert("end", f"    {text}\n", "dim")
        def sep():
            t.insert("end", "  " + "─" * 80 + "\n", "sep")

        # ===================================================================
        h1("━━━━━━━━━━━━━━━━━━━━━━━━━━  SETUP TAB  ━━━━━━━━━━━━━━━━━━━━━━━━━━")
        # -------------------------------------------------------------------
        h2("QUICK PRESETS")
        t.insert("end", """
    Clicking a preset (T300/5208, T300/914, AS4/3501-6) fills all ply property
    fields with published values and auto-computes the wave velocities.
    All fields remain editable — use a preset as a starting point and adjust
    any value you need.  Click 'Clear / New Material' to start from scratch.
""", "tip")

        sep()
        h2("PLY PROPERTIES  (left column)")
        row("E1  (GPa)",
            "Fibre-direction Young's modulus.  From prepreg data sheet.")
        note("  Typical CFRP: 100 – 180 GPa.  Matrix-dominated property E2 << E1.")
        row("E2  (GPa)",
            "Transverse Young's modulus.  From prepreg data sheet.")
        note("  Typical CFRP: 8 – 15 GPa.")
        row("G12  (GPa)",
            "In-plane shear modulus.  From prepreg data sheet.")
        note("  Typical CFRP: 4 – 8 GPa.")
        row("ν12  (dimensionless)",
            "Major Poisson's ratio.  From prepreg data sheet.")
        note("  Typical CFRP: 0.20 – 0.35.   ν21 = ν12 × E2/E1  (computed internally).")
        row("ρ  (kg/m³)",
            "Cured laminate density.  From data sheet or measured (mass / volume).")
        note("  Typical CFRP: 1 450 – 1 600 kg/m³.")
        row("Thickness  (mm)",
            "Total plate thickness.  Measure with calipers or count plies × ply thickness.")
        note("  Typical cured ply thickness: 0.125 – 0.25 mm.")
        note("  *** This value scales ALL dispersion results via fd = f × thickness. ***")

        sep()
        h2("WAVE VELOCITIES  (computed or entered manually)")
        t.insert("end", """
    Click  'Compute c_L / c_S / c_R (CLT)'  to calculate the three bulk wave
    speeds from the ply properties above using Classical Laminate Theory (CLT)
    invariants for a quasi-isotropic [0/45/-45/90]s layup.
    You may also type values directly if you have measured them ultrasonically.
""", "tip")
        row("c_L  (m/s)",
            "Longitudinal bulk wave speed.  Used as vp_max reference and CFL denominator.")
        note("  Typical CFRP: 5 000 – 10 000 m/s.")
        row("c_S  (m/s)",
            "Shear bulk wave speed.  Reference line on dispersion plots.")
        note("  Typical CFRP: 2 000 – 4 000 m/s.")
        row("c_R  (m/s)",
            "Rayleigh surface wave speed (optional).  Leave blank to skip.")
        note("  c_R ≈ c_S × (0.862 + 1.14·ν) / (1 + ν).  Shown as dashed line on plots.")

        sep()
        h2("SOLVER SETTINGS  (right column)")
        row("Symmetric modes",
            "How many S-modes to compute: S0, S1, S2 …  Default 5 is sufficient.")
        row("Antisymmetric modes",
            "How many A-modes to compute: A0, A1, A2 …  Default 5 is sufficient.")
        row("fd_max  (kHz·mm)",
            "Upper limit of the frequency-thickness axis.  = f_max(kHz) × thickness(mm).")
        note("  Example: 500 kHz on a 2 mm plate → fd_max = 1 000.  Default 5 000 is safe.")
        row("vp_max  (m/s)",
            "Upper phase-velocity search limit.  Keep above c_L.  Default 15 000 is safe.")
        row("fd_points",
            "Number of grid points along the fd axis.  200 = smooth enough for most uses.")
        row("vp_step  (m/s)",
            "Phase-velocity scan step.  50 is fine; decrease to 10–20 if modes are missing.")
        row("Material name",
            "Label used in plot titles and output file names.  Leave blank for 'custom'.")

        # ===================================================================
        h1("━━━━━━━━━━━━━━━━━━━━━━  DISPERSION TAB — OUTPUTS  ━━━━━━━━━━━━━━━━━━━━━━")
        t.insert("end", """
    The three plot buttons generate dispersion curves for the solved laminate.
    Use the matplotlib toolbar (bottom of the plot) to zoom, pan, and save PNGs.
""", "tip")
        h2("Phase Velocity  (vp  vs  fd)")
        t.insert("end", """
    X-axis: fd = frequency × thickness  [kHz·mm]
    Y-axis: phase velocity  [m/s]

    What it shows:
      The speed at which the wave crests travel.  Each curve is one mode.
      S-modes (symmetric) are shown in blue (default); A-modes in red.

    How to use:
      • Identify a flat region (low slope) on the mode you want — that is
        where dispersion is low and pulses travel without spreading.
      • Read off the fd value at your operating point to use in the Abaqus tab.
      • Cutoff-frequency markers show where each higher mode first appears.
""", "tip")
        h2("Group Velocity  (vg  vs  fd)")
        t.insert("end", """
    Y-axis: group velocity  [m/s]  — the speed at which energy (and the
    wave packet envelope) travels.

    How to use:
      • This is what you measure in a time-of-flight SHM experiment.
      • A flat region means less pulse distortion over distance.
      • Pick your operating frequency on a plateau of the group-velocity curve.
      • vg = d(ω)/dk = vp − λ · d(vp)/dλ
""", "tip")
        h2("Wave Number  (k  vs  fd)")
        t.insert("end", """
    Y-axis: wavenumber  k = ω / vp  [1/m]

    How to use:
      • Needed for analytical wave-structure calculations.
      • Slope of k(fd) ∝ 1/vp.  A steep slope means short wavelength (fine mesh needed).
""", "tip")

        # ===================================================================
        h1("━━━━━━━━━━━━━━━━━━━━━━  WAVE STRUCTURE TAB — OUTPUTS  ━━━━━━━━━━━━━━━━━━━")
        t.insert("end", """
    Shows the displacement profiles (u_x, u_z) through the plate thickness at
    a chosen fd value for one mode.  A grid of subplots is produced — one panel
    per fd value.
""", "tip")
        row("Mode  (dropdown)",
            "Which Lamb mode to plot (A0, S0, A1, S1 …).  Populated after SOLVE.")
        row("Rows / Cols",
            "Grid layout.  Rows × Cols = total number of fd snapshots shown.")
        row("fd values",
            "'auto' → evenly spaced fd values up to fd_max.  Or enter a comma-separated")
        note("  list, e.g.  50,100,200,500,1000  (must equal Rows × Cols).")
        t.insert("end", """
    Reading the plots:
      • Horizontal axis = normalised displacement amplitude.
      • Vertical axis   = position through thickness  (−h to +h, 0 = mid-plane).
      • u_x (in-plane) and u_z (out-of-plane) are plotted as separate curves.
      • Symmetric modes: u_z is symmetric, u_x is antisymmetric about the mid-plane.
      • Antisymmetric modes: opposite symmetry.
      • Use these plots to choose actuator/sensor position:
          – Stick a surface sensor where |u_z| is largest.
          – For a mode with zero u_z at the surface, use an embedded sensor.
""", "tip")

        # ===================================================================
        h1("━━━━━━━━━━━━━━━━━━━━━━━━  ANIMATION TAB  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        t.insert("end", """
    Generates a quiver-arrow animation showing how particles move as the wave
    passes through one wavelength.  Output: GIF + self-contained HTML file
    (opens automatically in your browser).
""", "tip")
        row("Mode",
            "Which Lamb mode to animate.  Populated after SOLVE.")
        row("fd  (kHz·mm)",
            "Operating fd value.  Must be within the solved fd range.")
        note("  Tip: use the fd of your Abaqus operating point for a consistent picture.")
        row("Frame interval  (ms)",
            "Time between animation frames.  Lower = faster playback.")
        note("  20–40 ms ≈ fast (good for high-frequency modes).")
        note("  80–150 ms ≈ slow (easier to follow low-frequency / long-wavelength modes).")
        t.insert("end", """
    Output files  (saved to results/):
      Mode_<mode>_fd<value>_<material>.gif   — portable animated GIF
      Mode_<mode>_fd<value>_<material>.html  — interactive HTML (no extra software needed)
""", "tip")

        # ===================================================================
        h1("━━━━━━━━━━━━━━━━━━━  ABAQUS PARAMETERS TAB — INPUTS  ━━━━━━━━━━━━━━━━━━━")
        t.insert("end", """
    This tab computes all mesh and time-stepping parameters needed to set up
    an Abaqus/Explicit simulation of Lamb wave propagation.
    Workflow:  1. SOLVE (Setup tab)  →  2. Fill inputs below  →  3. Click Compute.
""", "tip")
        row("Excitation frequency (kHz)",
            "Centre frequency of the tone burst applied to the actuator.")
        note("  Choose a frequency where the target mode has low dispersion (flat vg curve).")
        note("  Compute fd = freq × thickness and check the dispersion plot.")
        row("Elements per wavelength",
            "Minimum spatial resolution rule.  Le = λ / epw.")
        note("  Use ≥ 10 for engineering accuracy;  ≥ 20 for research.")
        row("Elements through thickness",
            "Depth resolution.  Min 2 for A0/S0;  4–8 for higher modes.")
        row("CFL safety factor",
            "Time-step = CFL × Le / (c_L × √3).  Use 0.9 (standard).")
        note("  Must be < 1.0 to keep the explicit solver stable.")
        row("Tone burst cycles",
            "Number of sine cycles in the Hanning-windowed excitation pulse.")
        note("  3–5 cycles: narrow bandwidth, good mode selectivity.  Common SHM choice.")
        row("Amplitude  (mm)",
            "Peak U2 (out-of-plane) displacement applied to the actuator node set.")
        note("  Keep small (0.01 – 0.1 mm) to ensure linear behaviour.")
        row("Total simulation time (µs)",
            "Run the simulation long enough for the wave to travel + reflect.")
        note("  Rule of thumb: t_sim > 2 × plate_length / group_velocity.")
        note("  Example: 300 mm plate, vg = 3 000 m/s → t > 200 µs → use 300 µs.")
        row("Plate L  /  W  (mm)",
            "In-plane dimensions of the Abaqus plate model.")
        note("  Match your physical specimen or use infinite-element boundaries.")
        row("Target mode",
            "Which Lamb mode the simulation is designed to excite (A0, S0 …).")
        note("  A0 is most common for low-frequency SHM.  S0 for high-speed inspection.")

        # ===================================================================
        h1("━━━━━━━━━━━━━━━━━━  ABAQUS PARAMETERS TAB — OUTPUTS  ━━━━━━━━━━━━━━━━━━")
        t.insert("end", """
    After clicking 'Compute Abaqus Parameters' the right panel shows a summary
    ready to copy into Abaqus or a simulation script.  Each block is explained below.
""", "tip")
        h2("OPERATING POINT")
        row("fd  (kHz·mm)",
            "freq (kHz) × thickness (mm).  Your position on the dispersion curve.")
        row("Phase velocity  vp  (m/s)",
            "Speed of wave crests at your fd.  Read from the dispersion curve.")
        row("Group velocity  vg  (m/s)",
            "Speed of the wave packet envelope.  Used to estimate arrival time.")
        row("Wavelength  λ  (mm)",
            "λ = vp / f.  Governs the required mesh size.")
        row("S0 reference values",
            "vp, vg, λ for the S0 mode at the same fd (for comparison).")

        h2("MESH  (element type C3D8R)")
        row("In-plane size  Le  (mm)",
            "Le = λ / epw.  Use this as the global seed in Abaqus Mesh.")
        note("  In Abaqus: Mesh → Seed Part → Approximate global size = Le.")
        row("Through thickness  (elements)",
            "Number of elements stacked through the plate thickness = eth.")
        note("  In Abaqus: partition the thickness into eth equal layers.")
        row("Elements X / Y",
            "Number of elements along each plate edge = ceil(plate_dim / Le).")
        row("Total elements  (est.)",
            "n_x × n_y × eth.  Guides you to estimate RAM and CPU time.")
        note("  Rule of thumb: each ~1 M elements needs ~1–2 GB RAM in Abaqus/Explicit.")

        h2("TIME STEPPING")
        row("dt  (s)",
            "Stable explicit time increment = CFL × Le / (c_L × √3).")
        note("  In Abaqus: Step → Time Increment → Fixed = dt  (or let Abaqus choose).")
        row("Total time  (µs)",
            "Total duration of the Abaqus Step.  Enter this as the Step Time.")
        row("Steps  (est.)",
            "t_sim / dt.  Total number of increments — governs wall-clock time.")

        h2("EXCITATION  (Hanning-windowed tone burst)")
        row("Frequency  fc  (kHz)",
            "Carrier frequency of the burst signal.")
        row("Cycles",
            "Number of sine cycles within the Hanning window.")
        row("Duration  t1  (µs)",
            "t1 = N_cycles / fc.  The signal is zero after t1.")
        row("Amplitude  A  (mm)",
            "Peak displacement applied at the actuator.")
        t.insert("end", """
    Equation  (apply as Abaqus Amplitude → Tabular or use the exported CSV):

        U2(t) = −A/2 · sin(2π·fc·t) · (1 − cos(2π·fc·t / N))   for 0 ≤ t ≤ t1
        U2(t) = 0                                                  for t > t1

    How to apply in Abaqus:
      1. Create an Amplitude → Tabular from the exported CSV file (time, value).
      2. Create a Boundary Condition → Displacement/Rotation on the actuator node set.
      3. Set U2 = 1.0 and reference the Amplitude definition.
      4. The actual amplitude scaling is already encoded in the CSV values.
""", "tip")

        h2("EXPORT BUTTONS")
        row("Dispersion + Op.Point  (PNG)",
            "Phase and group velocity curves with a star marking your operating fd.")
        note("  Use in your thesis / report to justify the chosen frequency.")
        row("Tone Burst  (PNG / CSV / FFT)",
            "Time-domain plot of the excitation signal, Abaqus-ready CSV amplitude table,")
        note("  and FFT spectrum verifying the narrowband content.")
        note("  Import the CSV as an Abaqus Amplitude → Tabular definition.")
        row("Summary Card  (PNG)",
            "A single-page image with all parameters — ideal for lab notes or a report.")
        row("Parameters  (TXT)",
            "Plain-text file with all values — easy to paste into a simulation script.")
        row("Export All",
            "Runs all four exports at once.")

        # ===================================================================
        h1("━━━━━━━━━━━━━━━━━━━━━━━━━━━━  EXPORT TAB  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        t.insert("end", """
    'Save All Results → TXT'  writes the full numerical dispersion data
    (phase velocity, group velocity, wave number vs fd) for every solved mode
    to a text file in the results/ folder.  This data can be loaded into MATLAB,
    Python, or Excel for further analysis.

    'Open Results Folder'  opens the results/ folder in Windows Explorer so you
    can access all saved PNGs, GIFs, HTMLs, CSVs, and TXTs directly.
""", "tip")

        t.config(state="disabled")

    # =======================================================================
    # COLOR PICKER
    # =======================================================================
    def _pick_color(self, which):
        init = self._sym_col if which == "sym" else self._anti_col
        res  = colorchooser.askcolor(
            color=init,
            title=f"Pick {'Symmetric' if which == 'sym' else 'Antisymmetric'} curve color")
        if res and res[1]:
            if which == "sym":
                self._sym_col = res[1]
                self._sym_btn.config(text=f"Sym:  {res[1]}")
            else:
                self._anti_col = res[1]
                self._anti_btn.config(text=f"Anti: {res[1]}")


# ===========================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()
