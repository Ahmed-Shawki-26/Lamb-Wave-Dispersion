# -*- coding: utf-8 -*-
"""
material_config.py
==================
Material property definitions for Lamb wave delamination detection project.

All three CFRP material systems:
  - T300/5208  (Oil & gas pressure vessels, pipelines)
  - T300/914   (Offshore structures, pressure vessels)
  - AS4/3501-6 (Chemical tanks, high-pressure vessels)

Unit system: mm, tonne (1e-9 kg), s, MPa
  - Moduli:  GPa × 1000 = MPa
  - Density: kg/m³ ÷ 1e12 = tonne/mm³

Sources: PROJECT_GUIDE.md (Tsai & Hahn, Daniel & Ishai, Barbero, NASA CR-187229)
"""

MATERIALS = {

    # =========================================================================
    # Material 1 — T300/5208
    # Fiber volume fraction: Vf = 0.62
    # Source: Tsai & Hahn (1980); Daniel & Ishai; Kriz & Stinchcomb (1979)
    # =========================================================================
    'T300_5208': {
        'display_name': 'T300/5208',
        'industry_use': 'Oil & gas pressure vessels, pipelines',

        # --- Ply-level elastic constants (transverse isotropic) ---
        # Stored in MPa (GPa × 1000)
        'E1':   132000.0,    # 132 GPa — Longitudinal modulus
        'E2':    10800.0,    #  10.8 GPa — Transverse modulus
        'E3':    10800.0,    # = E2 (transverse isotropy)
        'G12':    5650.0,    #   5.65 GPa — In-plane shear modulus
        'G13':    5650.0,    # = G12 (transverse isotropy)
        'G23':    3380.0,    #   3.38 GPa — Out-of-plane shear modulus
        'nu12':      0.24,   # Major Poisson's ratio
        'nu13':      0.24,   # = nu12 (transverse isotropy)
        'nu23':      0.59,   # Transverse Poisson's ratio

        # --- Density ---
        # 1520 kg/m³ → 1520 / 1e12 = 1.52e-9 tonne/mm³
        'density': 1.52e-9,
        'density_SI': 1520.0,  # kg/m³ (for reference only)

        # --- Ply thickness ---
        'ply_thickness': 0.125,  # mm

        # --- Phase 1 results (from Lamb-Wave-Dispersion/results) ---
        'A0_group_velocity': 1440.0,    # m/s at 50 kHz
        'A0_wavelength':       14.69,   # mm at 50 kHz
        'mesh_size':            1.47,    # mm (λ/10)
        'dt':                   1.13e-7, # s (CFL 0.9)

        # --- Equivalent laminate properties (Laminate Invariant Method) ---
        'Ex':  52200.0,   # 52.20 GPa in MPa
        'nuxy':    0.298,
        'Gxy': 20110.0,   # 20.11 GPa in MPa

        # --- Wave velocities ---
        'c_L': 6780.0,  # m/s — longitudinal
        'c_S': 3638.0,  # m/s — shear
        'c_R': 3370.0,  # m/s — Rayleigh

        # --- Tone burst CSV path (relative to project root) ---
        'tone_burst_csv': 'Lamb-Wave-Dispersion/results/T300_5208_1.0mm_cL6779/tone_burst_T300_5208_50kHz.csv',
    },

    # =========================================================================
    # Material 2 — T300/914
    # Fiber volume fraction: Vf = 0.60
    # Source: Barbero, Introduction to Composite Materials Design; Iliescu et al.
    # Note: G23 estimated via transverse isotropy formula
    # =========================================================================
    'T300_914': {
        'display_name': 'T300/914',
        'industry_use': 'Offshore structures, pressure vessels',

        # --- Ply-level elastic constants (transverse isotropic) ---
        'E1':   138000.0,    # 138 GPa
        'E2':    11000.0,    #  11.0 GPa
        'E3':    11000.0,    # = E2
        'G12':    5500.0,    #   5.5 GPa
        'G13':    5500.0,    # = G12
        'G23':    3930.0,    #   3.93 GPa (estimated)
        'nu12':      0.28,
        'nu13':      0.28,   # = nu12
        'nu23':      0.40,

        # --- Density ---
        # 1550 kg/m³ → 1.55e-9 tonne/mm³
        'density': 1.55e-9,
        'density_SI': 1550.0,

        # --- Ply thickness ---
        'ply_thickness': 0.125,  # mm

        # --- Phase 1 results ---
        'A0_group_velocity': 1448.0,
        'A0_wavelength':       14.78,
        'mesh_size':            1.48,
        'dt':                   1.11e-7,

        # --- Equivalent laminate properties ---
        'Ex':  54130.0,   # 54.13 GPa in MPa
        'nuxy':    0.307,
        'Gxy': 20720.0,   # 20.72 GPa in MPa

        # --- Wave velocities ---
        'c_L': 6922.0,
        'c_S': 3656.0,
        'c_R': 3390.0,

        # --- Tone burst CSV path ---
        'tone_burst_csv': 'Lamb-Wave-Dispersion/results/T300_914_1.0mm_cL6922/tone_burst_T300_914_50kHz.csv',
    },

    # =========================================================================
    # Material 3 — AS4/3501-6
    # Fiber volume fraction: Vf = 0.62
    # Source: Daniel & Ishai; NASA Contractor Report 187229 (1992)
    # Note: G12 = 5.61 GPa (Daniel & Ishai value — self-consistent set)
    # =========================================================================
    'AS4_3501_6': {
        'display_name': 'AS4/3501-6',
        'industry_use': 'Chemical tanks, high-pressure vessels',

        # --- Ply-level elastic constants (transverse isotropic) ---
        'E1':   148000.0,    # 148 GPa
        'E2':    10500.0,    #  10.5 GPa
        'E3':    10500.0,    # = E2
        'G12':    5610.0,    #   5.61 GPa
        'G13':    5610.0,    # = G12
        'G23':    3170.0,    #   3.17 GPa
        'nu12':      0.30,
        'nu13':      0.30,   # = nu12
        'nu23':      0.59,

        # --- Density ---
        # 1540 kg/m³ → 1.54e-9 tonne/mm³
        'density': 1.54e-9,
        'density_SI': 1540.0,

        # --- Ply thickness ---
        'ply_thickness': 0.125,  # mm

        # --- Phase 1 results ---
        'A0_group_velocity': 1475.0,
        'A0_wavelength':       15.03,
        'mesh_size':            1.50,
        'dt':                   1.09e-7,

        # --- Equivalent laminate properties ---
        'Ex':  57410.0,   # 57.41 GPa in MPa
        'nuxy':    0.308,
        'Gxy': 21950.0,   # 21.95 GPa in MPa

        # --- Wave velocities ---
        'c_L': 7163.0,
        'c_S': 3776.0,
        'c_R': 3502.0,

        # --- Tone burst CSV path ---
        'tone_burst_csv': 'Lamb-Wave-Dispersion/results/AS4_3501-6_1.0mm_cL7163/tone_burst_AS4_3501-6_50kHz.csv',
    },
}


# =============================================================================
# Layup configurations
# =============================================================================
LAYUP_CONFIGS = {
    'QI': {
        'display_name': 'Quasi-Isotropic [0/45/-45/90]s',
        'angles': [0, 45, -45, 90, 90, -45, 45, 0],   # symmetric: plies 1→8 (bottom→top)
        'num_plies': 8,
    },
    'UD': {
        'display_name': 'Unidirectional [0]8',
        'angles': [0, 0, 0, 0, 0, 0, 0, 0],
        'num_plies': 8,
    },
}


# =============================================================================
# Plate geometry (common to all cases)
# =============================================================================
PLATE = {
    'length': 300.0,     # mm (X direction)
    'width':  300.0,     # mm (Z direction)
    'thickness': 1.0,    # mm (Y direction)
    'num_plies': 8,
    'ply_thickness': 0.125,  # mm
}


# =============================================================================
# Delamination definitions (progressive staging)
# =============================================================================
DELAMINATIONS = {
    'stage1': {
        'display_name': 'Stage 1 — Baseline (20x20 mm, mid-depth)',
        'shape': 'rectangular',
        'size_x': 20.0,     # mm
        'size_z': 20.0,     # mm
        'center_x': 150.0,  # mm
        'center_z': 150.0,  # mm
        'interface': (4, 5),  # between ply 4 (bottom sub-laminate top) and ply 5 (top sub-laminate bottom)
        'depth_y': 0.5,      # mm from bottom surface (4 plies × 0.125)
    },
    'stage_multi': {
        'display_name': 'Stage Multi — 3 Damages (mid-depth)',
        'damages': [
            {'center_x': 100.0, 'center_z': 150.0, 'size_x': 20.0, 'size_z': 20.0},
            {'center_x': 200.0, 'center_z': 100.0, 'size_x': 25.0, 'size_z': 25.0},
            {'center_x': 200.0, 'center_z': 200.0, 'size_x': 15.0, 'size_z': 15.0},
        ],
        'interface': (4, 5),
        'depth_y': 0.5,
    },
}



# =============================================================================
# Wave excitation parameters
# =============================================================================
EXCITATION = {
    'frequency': 50000.0,    # Hz (50 kHz)
    'num_cycles': 3.5,
    'amplitude': 0.05,       # mm (U2 displacement)
    'excitation_duration': 70e-6,  # 70 µs = 70e-6 s
    'actuator_x': 75.0,      # mm
    'actuator_y': 0.0,       # mm (bottom surface)
    'actuator_z': 150.0,     # mm (plate center in Z)
    'actuator_radius': 3.0,  # mm — ring of nodes within this radius
    'direction': 'U2',       # Y-direction displacement
}


# =============================================================================
# Sensor array parameters
# =============================================================================
SENSORS = {
    'num_sensors': 7,
    'x_position': 225.0,  # mm — all sensors at same X
    'y_position': 0.0,    # mm — bottom surface
    'z_positions': [105.0, 120.0, 135.0, 150.0, 165.0, 180.0, 195.0],  # mm
    'spacing': 15.0,       # mm (≈ 1λ spacing)
    'output_variable': 'U2',
}


# =============================================================================
# Simulation step parameters
# =============================================================================
STEP = {
    'step_name': 'LambProp',
    'step_type': 'ExplicitDynamicsStep',
    'total_time': 300e-6,   # 300 µs = 300e-6 s
    'num_field_intervals': 200,   # for animation
    'num_history_intervals': 0,   # 0 = every increment
}
