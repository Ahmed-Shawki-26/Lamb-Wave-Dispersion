import sys
import os
import numpy as np

"""
COORDINATE CONVENTION:
    X (axis 0): 0° reference direction (typically fiber/warp)
    Y (axis 1): 90° direction (typically fill/weft)  
    Z (axis 2): Plate normal (layup/stacking direction)
    
Stiffness indices (Voigt): 1=XX, 2=YY, 3=ZZ, 4=YZ, 5=XZ, 6=XY
Index mapping in code (0-indexed): 0=XX, 1=YY, 2=ZZ, 3=YZ, 4=XZ, 5=XY
"""

# Add external path to allow importing the JanJaeken module
core_dir = os.path.dirname(__file__)
project_root = os.path.dirname(core_dir)
christoffel_path = os.path.join(project_root, 'external', 'christoffel')
if christoffel_path not in sys.path:
    sys.path.insert(0, christoffel_path)

try:
    from christoffel import Christoffel
except ImportError:
    # Fallback to a dummy if not found during development
    class Christoffel:
        def __init__(self, *args): pass
        def set_direction_cartesian(self, *args): pass
        def get_phase_velocity(self): return [0, 0, 0]
        def get_eigenvec(self): return np.eye(3)

def parse_layup(layup_str):
    """
    Parse complex layup strings into a list of angles.
    Supports:
      - Basic lists: [0/45/90]
      - Symmetry: [0/45]s -> [0, 45, 45, 0]
      - Subscripts/Repeats: [0_2/90_3] -> [0, 0, 90, 90, 90]
      - Group Repeats: [0/90]_2 -> [0, 90, 0, 90]
      - Combined: [0/45]_2s
    """
    import re
    s = layup_str.strip()
    if not s: return [0.0]

    # 1. Handle Symmetry at the very end
    is_symmetric = False
    if s.lower().endswith('s'):
        is_symmetric = True
        s = s[:-1]

    # 2. Extract global multiplier if exists, e.g., [0/90]_2 -> [0/90] and multiplier 2
    # Matches strings like "...]_3" or "...]3"
    global_mult = 1
    match_mult = re.search(r'\]_?(\d+)$', s)
    if match_mult:
        global_mult = int(match_mult.group(1))
        s = s[:match_mult.start()] + ']'

    # 3. Clean brackets
    s = s.replace('[', '').replace(']', '')

    # 4. Split into segments by common delimiters
    segments = re.split(r'[/,; ]+', s)
    
    angles = []
    for seg in segments:
        seg = seg.strip()
        if not seg: continue
        
        # Handle individual ply repeats like 0_3 or 0(3) or 03
        # Match "angle" and "count"
        # Case: 0_2 or 0-2 or 0^2 or +45_2
        match = re.match(r'([+-]?\d+\.?\d*)[_\^]?(?:\(?(\d+)\)?)?', seg)
        if match:
            angle = float(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            angles.extend([angle] * count)
        else:
            try:
                angles.append(float(seg))
            except ValueError:
                continue

    # 5. Apply global multiplier
    angles = angles * global_mult

    # 6. Apply symmetry
    if is_symmetric:
        angles = angles + angles[::-1]

    return angles if angles else [0.0]

def build_Cij_transversely_isotropic(E1, E2, G12, G23, nu12):
    """
    Build the 6×6 Voigt stiffness matrix for a transversely isotropic
    UD ply (fiber = axis 1, isotropy plane = 2-3).
    All inputs in Pa. Returns C in same units.
    """
    nu21 = nu12 * E2 / E1
    # Transverse isotropy constraint: nu23 = E2 / (2*G23) - 1
    nu23 = E2 / (2 * G23) - 1.0

    # Validate thermodynamic stability
    assert E1 > 0 and E2 > 0 and G12 > 0 and G23 > 0, "Moduli must be positive."
    assert -1 < nu23 < 1, f"Unphysical nu23={nu23:.3f}. Checked E2/(2*G23)-1."
    
    # Auxiliary determinant of the compliance sub-matrix (must be positive)
    Delta = (1 - nu23**2 - 2*nu12*nu21*(1 + nu23)) / (E1 * E2**2)
    assert Delta > 0, "Stiffness matrix not positive definite. Check material constants (nu12, E1, E2)."

    C11 = (1 - nu23**2)           / (E2**2 * Delta)
    C22 = (1 - nu12 * nu21)       / (E1 * E2 * Delta)
    C12 = nu21 * (1 + nu23)       / (E2**2 * Delta)
    C23 = (nu23 + nu12 * nu21)    / (E1 * E2 * Delta)
    C44 = G23
    C66 = G12

    C = np.zeros((6, 6))
    C[0,0] = C11
    C[1,1] = C22;  C[2,2] = C22      # C22 = C33
    C[0,1] = C12;  C[1,0] = C12      # C12
    C[0,2] = C12;  C[2,0] = C12      # C13 = C12
    C[1,2] = C23;  C[2,1] = C23      # C23
    C[3,3] = C44                       # G23
    C[4,4] = C66;  C[5,5] = C66      # G12 = G13

    return C

def bond_matrix(theta_deg):
    """6×6 Bond transformation matrix for rotation by theta about Z axis (Plate Normal)."""
    theta = np.radians(theta_deg)
    m, n = np.cos(theta), np.sin(theta)

    M = np.array([
        [m**2,    n**2,   0, 0, 0,   2*m*n    ],
        [n**2,    m**2,   0, 0, 0,  -2*m*n    ],
        [0,       0,      1, 0, 0,   0         ],
        [0,       0,      0, m, -n,  0         ],
        [0,       0,      0, n,  m,  0         ],
        [-m*n,    m*n,    0, 0, 0,   m**2-n**2 ]
    ])
    return M

def rotate_Cij(C, theta_deg):
    """
    Rotate 6×6 stiffness matrix by angle theta about Z (plate normal).
    Correctly handles stress/strain transformation using Reuter matrix.
    """
    T = bond_matrix(theta_deg)
    # Reuter matrix to handle engineering vs tensor shear strains
    R = np.diag([1, 1, 1, 2, 2, 2])
    R_inv = np.diag([1, 1, 1, 0.5, 0.5, 0.5])
    
    # Stiffness transforms as T_sigma @ C @ (T_eps)^T
    # where T_eps = R @ T_sigma @ R_inv
    T_eps = R @ T @ R_inv
    return T @ C @ T_eps.T

def homogenize_laminate_backus(E1, E2, G12, G23, nu12, angles_deg, thicknesses):
    """
    Compute effective 6×6 stiffness tensor using Backus (1962) averaging.
    Correct long-wavelength homogenization for layered media.
    """
    C_ply = build_Cij_transversely_isotropic(E1, E2, G12, G23, nu12)
    h_total = sum(thicknesses)
    
    C_list = []
    f_list = []
    for angle, t in zip(angles_deg, thicknesses):
        C_list.append(rotate_Cij(C_ply, angle))
        f_list.append(t / h_total)

    # Backus Averaging Pattern:
    # 1. Terms divided by C22: <1/C22>, <C02/C22>, <C12/C22>, <C25/C22>
    # 2. In-plane "Voigt" terms: <C00>, <C11>, <C12>, <C05>, <C15>, <C55>
    # 3. Correlation terms: <(Ci2*Cj2)/C22>
    
    # Out-of-plane denominator averaging
    inv_C22_avg = sum(f * 1.0/C[2,2] for f, C in zip(f_list, C_list))
    C33_eff = 1.0 / inv_C22_avg
    
    # Normalized interaction terms
    C13_norm_avg = sum(f * C[0,2]/C[2,2] for f, C in zip(f_list, C_list))
    C23_norm_avg = sum(f * C[1,2]/C[2,2] for f, C in zip(f_list, C_list))
    C36_norm_avg = sum(f * C[2,5]/C[2,2] for f, C in zip(f_list, C_list))
    
    # Effective C_i3 terms
    C13_eff = C13_norm_avg * C33_eff
    C23_eff = C23_norm_avg * C33_eff
    C36_eff = C36_norm_avg * C33_eff

    # Shear terms (C44, C45, C55) via 2x2 inverse averaging
    S_shear_avg = np.zeros((2, 2))
    for f, C in zip(f_list, C_list):
        S_shear_avg += f * np.linalg.inv(C[3:5, 3:5])
    C_shear_eff = np.linalg.inv(S_shear_avg)
    
    # In-plane terms (Ci,j for i,j in [0, 1, 5] -> XX, YY, XY)
    # Formula: C_ij_eff = <C_ij - (Ci2*Cj2)/C22> + (Ci3_eff * <Cj2/C22>)
    def backus_ij(idx_i, idx_j):
        """Backus correction for in-plane indices (0, 1, 5) only."""
        voigt_part = sum(f * (C[idx_i, idx_j] - (C[idx_i, 2]*C[idx_j, 2])/C[2,2]) for f, C in zip(f_list, C_list))
        # Ci3_eff is already ( <Ci2/C22> * C33_eff )
        # So we just need: voigt_part + Ci3_eff * <Cj2/C22>
        # Let's use the explicit indices: 0 -> C13, 1 -> C23, 5 -> C36
        norm_map = {0: C13_norm_avg, 1: C23_norm_avg, 5: C36_norm_avg}
        eff_map  = {0: C13_eff, 1: C23_eff, 5: C36_eff}
        return voigt_part + eff_map[idx_i] * norm_map[idx_j]

    C11_eff = backus_ij(0, 0)
    C22_eff = backus_ij(1, 1)
    C66_eff = backus_ij(5, 5)
    C12_eff = backus_ij(0, 1)
    C16_eff = backus_ij(0, 5)
    C26_eff = backus_ij(1, 5)

    C_eff = np.zeros((6, 6))
    # Direct assignments
    C_eff[0,0], C_eff[1,1], C_eff[2,2] = C11_eff, C22_eff, C33_eff
    C_eff[0,1] = C12_eff;  C_eff[1,0] = C12_eff
    C_eff[0,2] = C13_eff;  C_eff[2,0] = C13_eff
    C_eff[1,2] = C23_eff;  C_eff[2,1] = C23_eff
    
    # Shear block
    C_eff[3,3] = C_shear_eff[0,0]; C_eff[4,4] = C_shear_eff[1,1]
    C_eff[3,4] = C_shear_eff[0,1]; C_eff[4,3] = C_shear_eff[1,0]
    
    # Extension-Shear and pure SH terms
    C_eff[5,5] = C66_eff
    C_eff[0,5] = C16_eff;  C_eff[5,0] = C16_eff
    C_eff[1,5] = C26_eff;  C_eff[5,1] = C26_eff
    C_eff[2,5] = C36_eff;  C_eff[5,2] = C36_eff
    
    return C_eff

def extract_cL_cS_for_direction(C_eff, rho, theta_deg):
    """
    Extract c_L and c_S for a given in-plane propagation direction
    from the homogenized 6x6 stiffness matrix using the Christoffel solver.
    Uses sagittal plane polarization logic to identify modes.
    Note: Christoffel module (JanJaeken) transposes eigenvectors internally 
    to rows: pol[i] is the vector for mode i.
    """
    # Christoffel module expects stiffness in GPa and density in kg/m3.
    # Internally it calculates v based on (C/rho) * 1000 if input is GPa.
    # Output v is in km/s.
    C_GPa = C_eff / 1e9
    
    chris = Christoffel(C_GPa, rho)
    
    rad = np.radians(theta_deg)
    direction = np.array([np.cos(rad), np.sin(rad), 0.0])
    chris.set_direction_cartesian(direction)
    
    v = chris.get_phase_velocity()  # km/s
    pol = chris.get_eigenvec()      # Row indexing (verified in library source)
    
    plate_normal = np.array([0.0, 0.0, 1.0])
    sagittal_normal = np.cross(direction, plate_normal)
    sn_norm = np.linalg.norm(sagittal_normal)
    if sn_norm > 1e-12: sagittal_normal /= sn_norm
    
    # Identify modes using polarization vectors (pol[i])
    p_scores  = [abs(np.dot(pol[i], direction)) for i in range(3)]
    sh_scores = [abs(np.dot(pol[i], sagittal_normal)) for i in range(3)]
    
    idx_p  = np.argmax(p_scores)
    idx_sh = np.argmax(sh_scores)
    idx_sv = 3 - idx_p - idx_sh   # remaining mode = qSV (through-thickness shear)

    # Velocity in m/s
    # c_S must be qSV (interlaminar shear, plate-normal polarized) for Lamb wave P-SV coupling
    c_L = v[idx_p]  * 1000.0
    c_S = v[idx_sv] * 1000.0

    return c_L, c_S

def extract_velocities_polar(C_eff, rho, thetas_deg):
    """
    Compute cL, cS for an array of in-plane angles efficiently
    by reusing the Christoffel object and pre-calculating common values.
    """
    C_GPa = C_eff / 1e9
    chris = Christoffel(C_GPa, rho)
    
    plate_normal = np.array([0.0, 0.0, 1.0])
    cL_list = []
    cS_list = []
    
    for theta in thetas_deg:
        rad = np.radians(theta)
        direction = np.array([np.cos(rad), np.sin(rad), 0.0])
        chris.set_direction_cartesian(direction)
        
        v = chris.get_phase_velocity()
        pol = chris.get_eigenvec()
        
        sagittal_normal = np.cross(direction, plate_normal)
        sn_norm = np.linalg.norm(sagittal_normal)
        if sn_norm > 1e-12: sagittal_normal /= sn_norm
        
        p_scores  = [abs(np.dot(pol[i], direction)) for i in range(3)]
        sh_scores = [abs(np.dot(pol[i], sagittal_normal)) for i in range(3)]
        
        idx_p  = np.argmax(p_scores)
        idx_sh = np.argmax(sh_scores)
        
        cL_list.append(v[idx_p] * 1000.0)
        cS_list.append(v[idx_sh] * 1000.0)
        
    return np.array(cL_list), np.array(cS_list)

def get_christoffel_full(C_eff, rho, theta_deg):
    """
    Return all Christoffel outputs at one in-plane propagation angle.
    Identifies quasi-P (qP), quasi-SH, and quasi-SV modes by polarization.

    Parameters
    ----------
    C_eff     : 6×6 numpy array in Pa  (Voigt notation)
    rho       : float, density in kg/m³
    theta_deg : float, in-plane angle from X-axis in degrees

    Returns
    -------
    dict with keys:
        c_qP, c_qSH, c_qSV    — phase velocities (m/s)
        vg_qP, vg_qSH, vg_qSV — group velocity magnitudes (m/s)
        pf_qP, pf_qSH, pf_qSV — powerflow angles (degrees)
        iso_P, iso_S           — Voigt-isotropic reference velocities (m/s)
        idx_qP, idx_qSH, idx_qSV — mode indices in sorted eigenvalue array
    """
    C_GPa = C_eff / 1e9
    chris = Christoffel(C_GPa, rho)

    rad = np.radians(theta_deg)
    direction = np.array([np.cos(rad), np.sin(rad), 0.0])
    chris.set_direction_cartesian(direction)

    phase_vel     = chris.get_phase_velocity()   # km/s, shape (3,), sorted low→high
    group_vel_abs = chris.get_group_abs()         # km/s, shape (3,)
    powerflow_rad = chris.get_powerflow()         # rad,  shape (3,), UNSIGNED
    eigenvec      = chris.get_eigenvec()          # shape (3,3), rows = polarization vectors
    group_dirs    = chris.get_group_dir()         # shape (3,3), rows = unit group direction vectors

    plate_normal = np.array([0.0, 0.0, 1.0])
    sagittal_normal = np.cross(direction, plate_normal)
    sn_norm = np.linalg.norm(sagittal_normal)
    if sn_norm > 1e-12:
        sagittal_normal /= sn_norm

    p_scores  = [abs(np.dot(eigenvec[i], direction))       for i in range(3)]
    sh_scores = [abs(np.dot(eigenvec[i], sagittal_normal)) for i in range(3)]

    idx_qP  = int(np.argmax(p_scores))
    idx_qSH = int(np.argmax(sh_scores))
    idx_qSV = 3 - idx_qP - idx_qSH

    def _signed_pf(mode_idx):
        """Return powerflow angle with sign: +ve = CCW rotation from phase direction."""
        pf_unsigned = np.degrees(powerflow_rad[mode_idx])
        gd = group_dirs[mode_idx]
        # cross(phase_dir, group_dir) z-component: +ve → group is CCW from phase
        cross_z = direction[0] * gd[1] - direction[1] * gd[0]
        return pf_unsigned * (1.0 if cross_z >= 0 else -1.0)

    return {
        'c_qP':   phase_vel[idx_qP]  * 1000.0,
        'c_qSH':  phase_vel[idx_qSH] * 1000.0,
        'c_qSV':  phase_vel[idx_qSV] * 1000.0,
        'vg_qP':  group_vel_abs[idx_qP]  * 1000.0,
        'vg_qSH': group_vel_abs[idx_qSH] * 1000.0,
        'vg_qSV': group_vel_abs[idx_qSV] * 1000.0,
        'pf_qP':  np.degrees(powerflow_rad[idx_qP]),
        'pf_qSH': np.degrees(powerflow_rad[idx_qSH]),
        'pf_qSV': np.degrees(powerflow_rad[idx_qSV]),
        'pf_qP_signed':  _signed_pf(idx_qP),
        'pf_qSH_signed': _signed_pf(idx_qSH),
        'pf_qSV_signed': _signed_pf(idx_qSV),
        'iso_P':  chris.get_isotropic_P() * 1000.0,
        'iso_S':  chris.get_isotropic_S() * 1000.0,
        'idx_qP':  idx_qP,
        'idx_qSH': idx_qSH,
        'idx_qSV': idx_qSV,
    }


def compute_directional_dispersion(C_eff, rho, thickness_mm, fd_max=5000,
                                    thetas_deg=None, nmodes_sym=3,
                                    nmodes_antisym=3, vp_max=15000,
                                    fd_points=200, vp_step=50):
    """
    Compute direction-dependent Lamb wave dispersion using the
    Christoffel-guided quasi-isotropic approximation.

    For each propagation angle theta:
        1. Extract c_L(theta) and c_S(theta) from the Christoffel equation.
        2. Run the classical Rayleigh-Lamb solver at those velocities.
        3. Record A0 and S0 phase/group velocity curves.

    Note: This is an engineering approximation — exact for isotropic plates,
    good for quasi-isotropic laminates, approximate for strongly anisotropic
    layups (exact treatment requires Transfer Matrix Method).

    Parameters
    ----------
    C_eff        : 6x6 numpy array in Pa (Backus-averaged laminate stiffness)
    rho          : float, density in kg/m^3
    thickness_mm : float, plate thickness in mm
    fd_max       : float, max frequency-thickness product (kHz*mm)
    thetas_deg   : array-like of angles in degrees (default 0-360 in 10-deg steps)
    nmodes_sym   : int, number of symmetric Lamb modes to solve
    nmodes_antisym : int, number of antisymmetric Lamb modes to solve
    vp_max       : float, upper phase-velocity search limit (m/s)
    fd_points    : int, number of fd grid points
    vp_step      : int, phase-velocity bisection step (m/s)

    Returns
    -------
    dict with keys:
        'thetas'  : array of angles (deg)
        'fd'      : 1-D fd array (kHz*mm)
        'results' : list of per-angle dicts, each containing
                      'theta', 'c_L', 'c_S',
                      'pf_qP'  : qP  powerflow angle (deg) — steering estimate for S0 mode
                      'pf_qSH' : qSH powerflow angle (deg)
                      'pf_qSV' : qSV powerflow angle (deg) — steering estimate for A0 mode
                      'A0_vp', 'A0_vg', 'S0_vp', 'S0_vg'  (arrays over fd)

    Physical note on steering angles
    ---------------------------------
    For anisotropic plates (Neau et al. 2001), group velocity travels in a DIFFERENT
    direction than phase velocity. The steering angle (powerflow angle) is:
      - A0 mode: approaches qSV bulk powerflow angle at low fd
      - S0 mode: approaches qP  bulk powerflow angle at low fd
    The exact Lamb-wave steering angle is also frequency-dependent, but the bulk-wave
    approximation stored here is a valid first-order estimate at low fd.
    """
    from .lambwaves import Lamb

    if thetas_deg is None:
        thetas_deg = np.arange(0, 361, 10)

    fd_arr = np.linspace(0.1, fd_max, fd_points)
    all_results = []

    def _get_mode(spline_dict, key, fd):
        """Return spline values for a mode key; NaN where mode is undefined or out of range."""
        if key not in spline_dict:
            return np.full(len(fd), np.nan, dtype=float)
        spl = spline_dict[key]
        result = np.full(len(fd), np.nan, dtype=float)
        try:
            x_min = float(spl.x[0]) if hasattr(spl, 'x') else fd[0]
            x_max = float(spl.x[-1]) if hasattr(spl, 'x') else fd[-1]
            mask = (fd >= x_min) & (fd <= x_max)
            if mask.any():
                result[mask] = spl(fd[mask])
        except Exception:
            pass
        return result

    for theta in thetas_deg:
        # Use get_christoffel_full: one call gives c_L, c_S AND all powerflow angles
        chris_res = get_christoffel_full(C_eff, rho, float(theta))
        c_L = chris_res['c_qP']   # quasi-P longitudinal velocity (m/s)
        c_S = chris_res['c_qSV']  # quasi-SV through-thickness shear (m/s) — correct for P-SV Lamb coupling

        nan_arr = np.full(len(fd_arr), np.nan)
        try:
            lmb = Lamb(
                thickness      = thickness_mm,
                nmodes_sym     = nmodes_sym,
                nmodes_antisym = nmodes_antisym,
                fd_max         = fd_max,
                vp_max         = vp_max,
                c_L            = c_L,
                c_S            = c_S,
                fd_points      = fd_points,
                vp_step        = vp_step,
            )
            A0_vp = _get_mode(lmb.vp_antisym, 'A0', fd_arr)
            A0_vg = _get_mode(lmb.vg_antisym, 'A0', fd_arr)
            S0_vp = _get_mode(lmb.vp_sym,     'S0', fd_arr)
            S0_vg = _get_mode(lmb.vg_sym,     'S0', fd_arr)
        except Exception:
            A0_vp = A0_vg = S0_vp = S0_vg = nan_arr

        entry = {
            'theta':  float(theta),
            'c_L':    c_L,
            'c_S':    c_S,
            # Bulk-wave powerflow (steering) angles from Christoffel (unsigned and signed)
            'pf_qP':         chris_res['pf_qP'],          # deg unsigned — S0 steering estimate
            'pf_qSH':        chris_res['pf_qSH'],         # deg unsigned
            'pf_qSV':        chris_res['pf_qSV'],         # deg unsigned — A0 steering estimate
            'pf_qP_signed':  chris_res['pf_qP_signed'],   # deg signed (CCW = positive)
            'pf_qSH_signed': chris_res['pf_qSH_signed'],  # deg signed
            'pf_qSV_signed': chris_res['pf_qSV_signed'],  # deg signed
            # Lamb wave dispersion curves (arrays over fd_arr)
            'A0_vp': A0_vp,
            'A0_vg': A0_vg,
            'S0_vp': S0_vp,
            'S0_vg': S0_vg,
        }
        all_results.append(entry)

    return {'thetas': np.array(thetas_deg), 'fd': fd_arr, 'results': all_results}


def validate_christoffel_convention():
    """
    Verify the eigenvector convention (Row vs Column) of the Christoffel library.
    Carbon Fiber UD qP along fiber should align with the fiber direction.
    """
    # Isotropic material for base test: any direction, qP is parallel
    lam, mu = 10.0, 5.0 # GPa
    C_iso = np.zeros((6,6))
    C_iso[0,0] = C_iso[1,1] = C_iso[2,2] = lam + 2*mu
    C_iso[0,1] = C_iso[0,2] = C_iso[1,2] = lam
    C_iso[1,0] = C_iso[2,0] = C_iso[2,1] = lam
    C_iso[3,3] = C_iso[4,4] = C_iso[5,5] = mu
    
    try:
        chris = Christoffel(C_iso, 1000.0)
        direction = np.array([1.0, 0.0, 0.0])
        chris.set_direction_cartesian(direction)
        v = chris.get_phase_velocity()
        pol = chris.get_eigenvec()
        idx_p = np.argmax(v)
        p_dot = abs(np.dot(pol[idx_p], direction))
        if p_dot < 0.9:
            print("[WARNING] Christoffel core indexing mismatch. Expected pol[i].")
        return p_dot > 0.9
    except:
        return False
