import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import minimize_scalar

def build_tensor_from_voigt(C_voigt):
    """
    Converts a 6x6 Voigt stiffness matrix into a 3x3x3x3 tensor.
    Voigt index mapping (0-indexed):
    11->0, 22->1, 33->2, 23->3, 13->4, 12->5
    """
    voigt_map = {
        (0,0): 0, (1,1): 1, (2,2): 2,
        (1,2): 3, (2,1): 3,
        (0,2): 4, (2,0): 4,
        (0,1): 5, (1,0): 5
    }
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    v1 = voigt_map.get((i, j))
                    v2 = voigt_map.get((k, l))
                    if v1 is not None and v2 is not None:
                        C[i, j, k, l] = C_voigt[v1, v2]
    return C

class AnisotropicGMM:
    def __init__(self, C_voigt, rho, thickness_m):
        """
        Global Matrix Method (GMM) Solver for an arbitrary single anisotropic layer.
        Parameters:
            C_voigt : 6x6 numpy array of stiffnesses (Pa)
            rho     : float, density (kg/m^3)
            thickness_m : float, plate thickness in meters
        """
        self.C_voigt = C_voigt
        self.C = build_tensor_from_voigt(C_voigt)
        self.rho = rho
        self.h = thickness_m

    def compute_partial_waves(self, vp, theta_deg):
        """
        Solves Christoffel equation to find 6 partial waves for a given phase velocity (vp).
        Returns:
            alpha : 1D array (6,), vertical wavenumbers divided by horizontal wavenumber k
            U     : 2D array (3, 6), displacement polarization vectors
            V     : 2D array (3, 6), traction amplitude vectors (normalized by ik)
        """
        theta = np.radians(theta_deg)
        l1 = np.cos(theta)
        l2 = np.sin(theta)

        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        C_mat = np.zeros((3, 3))

        for i in range(3):
            for k in range(3):
                A[i, k] = (self.C[i, 0, k, 0] * l1*l1 +
                           self.C[i, 0, k, 1] * l1*l2 +
                           self.C[i, 1, k, 0] * l2*l1 +
                           self.C[i, 1, k, 1] * l2*l2)
                B[i, k] = (self.C[i, 0, k, 2] * l1 +
                           self.C[i, 1, k, 2] * l2 +
                           self.C[i, 2, k, 0] * l1 +
                           self.C[i, 2, k, 1] * l2)
                C_mat[i, k] = self.C[i, 2, k, 2]

        A_mod = A - self.rho * (vp**2) * np.eye(3)
        C_inv = np.linalg.inv(C_mat)

        H = np.zeros((6, 6))
        H[0:3, 3:6] = np.eye(3)
        H[3:6, 0:3] = -C_inv @ A_mod
        H[3:6, 3:6] = -C_inv @ B

        alpha, evecs = np.linalg.eig(H)
        U = evecs[0:3, :]  # top 3 rows are U_k

        V = np.zeros((3, 6), dtype=np.complex128)
        for q in range(6):
            for i in range(3):
                V[i, q] = sum(
                    self.C[2, i, k, 0] * l1 * U[k, q] +
                    self.C[2, i, k, 1] * l2 * U[k, q] +
                    self.C[2, i, k, 2] * alpha[q] * U[k, q]
                    for k in range(3)
                )

        return alpha, U, V

    def compute_determinant(self, fd, vp, theta_deg, alpha, U):
        """
        Builds the 6x6 characteristic GMM matrix and returns its smallest singular value (normalized).
        Roots of this condition match the Lamb wave modes. 
        """
        theta = np.radians(theta_deg)
        l1, l2 = np.cos(theta), np.sin(theta)
        
        gamma = 2.0 * np.pi * fd / vp
        M = np.zeros((6, 6), dtype=np.complex128)
        
        for q in range(6):
            a_q = alpha[q]
            U_q = U[:, q]
            
            # Calculate stress amplitudes (T_3j / ik)
            T = np.zeros(3, dtype=np.complex128)
            for i in range(3):
                T[i] = (self.C[2, i, 0, 0] * l1 * U_q[0] + 
                        self.C[2, i, 1, 0] * l2 * U_q[0] +
                        self.C[2, i, 0, 1] * l1 * U_q[1] +
                        self.C[2, i, 1, 1] * l2 * U_q[1] +
                        self.C[2, i, 2, 0] * a_q * U_q[0] +
                        self.C[2, i, 2, 1] * a_q * U_q[1] +
                        self.C[2, i, 0, 2] * l1 * U_q[2] +
                        self.C[2, i, 1, 2] * l2 * U_q[2] +
                        self.C[2, i, 2, 2] * a_q * U_q[2])

            # Exponents for top (-h/2) and bottom (+h/2)
            arg = gamma * a_q / 2.0
            phi_top = np.exp(-1j * arg)
            phi_bot = np.exp( 1j * arg)
            
            # Divide both by the larger exponential term to keep values <= 1
            # Scaling factor is exp(|Im(arg)|)
            scale = np.exp(-np.abs(np.imag(arg)))
            phi_top *= scale
            phi_bot *= scale
                
            M[0:3, q] = T * phi_top
            M[3:6, q] = T * phi_bot
            
        # Normalize columns for better numerical conditioning
        for q in range(6):
            norm = np.linalg.norm(M[:, q])
            if norm > 1e-20:
                M[:, q] /= norm
            
        # Return reciprocal condition number (smallest singular value / largest)
        s = np.linalg.svd(M, compute_uv=False)
        return s[-1] / s[0] if s[0] > 0 else 1.0

    def solve_dispersion(self, theta_deg, f_min_khz, f_max_khz, num_f, vp_array):
        """
        Sweeps through phase velocities and finds frequencies where Lamb modes exist.
        Returns array of roots: [vp (m/s), fd (Hz*m), vg_mag (m/s), steering_angle (deg)]
        """
        # Convert f limits from kHz to Hz
        fd_array = np.linspace(f_min_khz * 1e3 * self.h, f_max_khz * 1e3 * self.h, num_f)
        
        roots_vp = []
        roots_fd = []
        roots_vg = []
        roots_steering = []
        roots_symmetry = [] # 0 for Anti, 1 for Sym
        
        for vp in vp_array:
            try:
                alpha, U, V = self.compute_partial_waves(vp, theta_deg)
            except (np.linalg.LinAlgError, ValueError):
                continue

            # Evaluate determinant across the entire frequency range for this VP
            det_curve = np.array([self.compute_determinant(fd, vp, theta_deg, alpha, U) for fd in fd_array])
            
            # Roots manifest as massive destructive cancellation (dips)
            for i in range(1, len(det_curve) - 1):
                if det_curve[i] < det_curve[i-1] and det_curve[i] < det_curve[i+1]:
                    # Refine using simple bounded bracket
                    res = minimize_scalar(
                        self.compute_determinant, 
                        args=(vp, theta_deg, alpha, U), 
                        bounds=(fd_array[i-1], fd_array[i+1]),
                        method='bounded'
                    )
                    
                    if res.success and res.fun < 0.05: 
                        exact_fd = res.x
                        
                        # Only keep strictly positive frequency roots
                        if exact_fd <= 0:
                            continue
                            
                        # Compute symmetry and group velocity
                        # Evaluate null space of M at the root
                        gamma = 2.0 * np.pi * exact_fd / vp
                        M = np.zeros((6, 6), dtype=np.complex128)
                        for q in range(6):
                            phi_top = np.exp(-1j * gamma * alpha[q] / 2.0)
                            phi_bot = np.exp( 1j * gamma * alpha[q] / 2.0)
                            scale = np.exp(-np.abs(np.imag(gamma * alpha[q] / 2.0)))
                            # Stress amplitudes (T_3j / ik) T are calculated here
                            T = np.zeros(3, dtype=np.complex128)
                            for i in range(3):
                                T[i] = (self.C[2, i, 0, 0] * np.cos(np.radians(theta_deg)) * U[0, q] + 
                                        self.C[2, i, 1, 0] * np.sin(np.radians(theta_deg)) * U[0, q] +
                                        self.C[2, i, 0, 1] * np.cos(np.radians(theta_deg)) * U[1, q] +
                                        self.C[2, i, 1, 1] * np.sin(np.radians(theta_deg)) * U[1, q] +
                                        self.C[2, i, 2, 0] * alpha[q] * U[0, q] +
                                        self.C[2, i, 2, 1] * alpha[q] * U[1, q] +
                                        self.C[2, i, 0, 2] * np.cos(np.radians(theta_deg)) * U[2, q] +
                                        self.C[2, i, 1, 2] * np.sin(np.radians(theta_deg)) * U[2, q] +
                                        self.C[2, i, 2, 2] * alpha[q] * U[2, q])
                            M[0:3, q] = T * phi_top * scale
                            M[3:6, q] = T * phi_bot * scale
                        
                        U_null, S_null, Vh_null = np.linalg.svd(M)
                        W = Vh_null[-1, :] # Null space weights for partial waves
                        
                        # Displacement at Top and Bottom
                        Uz_top = sum(W[q] * U[2, q] * np.exp(-1j * gamma * alpha[q] / 2.0) for q in range(6))
                        Uz_bot = sum(W[q] * U[2, q] * np.exp( 1j * gamma * alpha[q] / 2.0) for q in range(6))
                         
                        # Parity: Uz_top == Uz_bot -> Antisymmetric (Bending), Uz_top == -Uz_bot -> Symmetric (Stretch)
                        # We check the phase of the ratio
                        if np.abs(Uz_top) > 1e-15:
                            ratio = Uz_bot / Uz_top
                            # If ratio is ~1 (0 deg phase), it's Antisymmetric
                            # If ratio is ~-1 (180 deg phase), it's Symmetric
                            symmetry = 1 if np.abs(np.angle(ratio, deg=True)) > 90 else 0
                        else:
                            symmetry = 0 # Fallback
                        
                        vg_x, vg_y = self.compute_group_velocity(exact_fd, vp, theta_deg, alpha, U, V)
                        vg_mag = np.sqrt(vg_x**2 + vg_y**2)
                        
                        # Steering angle calculated via group velocity vector phase vs theta
                        steering_angle = np.degrees(np.arctan2(vg_y, vg_x)) - theta_deg
                        # Normalize steering to [-180, 180]
                        steering_angle = (steering_angle + 180) % 360 - 180
                        
                        roots_vp.append(vp)
                        roots_fd.append(exact_fd)
                        roots_vg.append(vg_mag)
                        roots_steering.append(steering_angle)
                        roots_symmetry.append(symmetry)
        
        return np.column_stack((roots_vp, roots_fd, roots_vg, roots_steering, roots_symmetry)) if len(roots_vp) > 0 else np.array([])

    def compute_group_velocity(self, fd, vp, theta_deg, alpha, U, V):
        """
        Analytically integrates the exact power flow and energy density over the plate thickness 
        using the null space of the GMM characteristic matrix.
        Returns exact 2D Group Velocity Vector (Vg_x, Vg_y).
        """
        theta = np.radians(theta_deg)
        l1, l2 = np.cos(theta), np.sin(theta)
        gamma = 2.0 * np.pi * fd / vp
        k = gamma / self.h
        omega = 2.0 * np.pi * (fd / self.h)
        
        M = np.zeros((6, 6), dtype=np.complex128)
        for q in range(6):
            a_R, a_I = np.real(alpha[q]), np.imag(alpha[q])
            exp_top = np.exp(-1j * gamma * a_R / 2) * np.exp( gamma * a_I / 2 - gamma * abs(a_I) / 2)
            exp_bot = np.exp( 1j * gamma * a_R / 2) * np.exp(-gamma * a_I / 2 - gamma * abs(a_I) / 2)
            M[0:3, q] = V[:, q] * exp_top
            M[3:6, q] = V[:, q] * exp_bot
            
        _, _, Vh = np.linalg.svd(M)
        W_scaled = Vh[-1, :].conj()
        
        # Calculate integration coupling factors J_pq to avoid all exponent overflow
        J = np.zeros((6, 6), dtype=np.complex128)
        for p in range(6):
            for q in range(6):
                W_p = W_scaled[p]
                W_q_star = np.conj(W_scaled[q])
                
                c_pq = 1j * gamma * (alpha[p] - np.conj(alpha[q]))
                d_pq = - (gamma / 2.0) * (abs(np.imag(alpha[p])) + abs(np.imag(alpha[q])))
                
                if abs(c_pq) < 1e-10:
                    int_factor = self.h * np.exp(d_pq)
                else:
                    # e^{c_pq/2} and e^{-c_pq/2} safely combined with d_pq
                    term1 = np.exp(d_pq + c_pq / 2.0)
                    term2 = np.exp(d_pq - c_pq / 2.0)
                    int_factor = (self.h / c_pq) * (term1 - term2)
                
                J[p, q] = W_p * W_q_star * int_factor
                
        # Calculate Power vector P_x, P_y
        P_x, P_y = 0.0, 0.0
        E_tot = 0.0
        
        n_p = np.zeros((3, 6), dtype=np.complex128)
        n_p[0, :] = l1
        n_p[1, :] = l2
        n_p[2, :] = alpha
        
        for p in range(6):
            for q in range(6):
                # Time-averaged Kinetic Energy Density integral
                # e_kin = 0.5 * rho * |u_dot|^2 -> over a cycle, we get 0.25 * rho * omega^2 * u*u
                # Total energy e_tot = 2 * e_kin = 0.5 * rho * omega^2 * u*u
                U_dot = np.dot(U[:, p], np.conj(U[:, q]))
                E_tot += 0.5 * self.rho * (omega**2) * J[p, q] * U_dot
                
                # Power flow flux P_i = -0.5 * Re( sigma_ij * dot{u}_j^* )
                # dot{u} = -i * omega * u  => dot{u}^* = +i * omega * u_star
                # P_i = -0.5 * Re( sigma_ij * i * omega * u_star )
                # sigma_ij = i * k * sum(C_ijkl * n_l * U_k)
                for j in range(3):
                    # T_ij_p is already sigma_ij / (i*k)
                    # so stress is 1j * k * C_ijkl * n_l * U_k
                    T1j = sum(1j * k * self.C[0, j, k_l, l_l] * n_p[l_l, p] * U[k_l, p] for k_l in range(3) for l_l in range(3))
                    T2j = sum(1j * k * self.C[1, j, k_l, l_l] * n_p[l_l, p] * U[k_l, p] for k_l in range(3) for l_l in range(3))
                    
                    P_x += -0.5 * (1j * omega) * J[p, q] * T1j * np.conj(U[j, q])
                    P_y += -0.5 * (1j * omega) * J[p, q] * T2j * np.conj(U[j, q])
        
        P_x, P_y, E_tot = np.real(P_x), np.real(P_y), np.real(E_tot)
        
        if E_tot < 1e-30:
            return 0.0, 0.0
        
        return P_x / E_tot, P_y / E_tot
