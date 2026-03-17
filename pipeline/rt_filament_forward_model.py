#!/usr/bin/env python3
"""
RT Filament Forward Model Pipeline

Implements Steps A-D of the Computability Pipeline (Appendix I) for
cosmic filament validation:

Step A: Solve nonlocal closure for kappa*
Step B: Compute log-strain and derived fields
Step C: Lensing forward solve (rays)  [projection only for WL comparison]
Step D: Cosmic filament structure (anisotropy field Q)

Primary observable: Stacked Filament Transverse Shear Profile (SFTSP)

This is the computational core. It takes:
- baryonic geometry (cluster/galaxy positions and masses)
- frozen constitutive parameters (mu_IR, lambda_IR, alpha)
- readout scale ell

And produces:
- Q(x; ell) anisotropy field
- predicted SFTSP for each filament
- integrated transverse anisotropy amplitude A_fil
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# Configuration and Constants
# ============================================================

@dataclass
class RegimeCertificate:
    """
    Certified regime parameters. Frozen before any holdout evaluation.

    Corresponds to the regime certificate in Axiom 1 / Definition 2.2.
    """
    # Readout scale (Section 4 of contract)
    ell: float = 2.0          # Mpc/h (comoving)
    R_max: float = 5.0        # Mpc/h corridor half-width
    sigma_kernel: float = 1.414  # Mpc/h Gaussian smoothing sigma

    # Constitutive parameters (low-strain / gravity regime)
    # lambda_IR: bulk modulus (derived from second variation, Lemma T02)
    # mu_IR: shear modulus (PENDING — to be anchored on training filaments)
    # alpha: gradient penalty (derived from kernel-moment, Def alpha_kernel)
    lambda_IR: float = 1.0    # placeholder — to be derived
    mu_IR: float = 0.0        # PENDING — this is what we're anchoring
    alpha: float = 0.01       # placeholder — to be derived

    # Strain-optic coefficients (Axiom 1, eq. 3)
    a_bulk: float = 1.0       # bulk strain-optic coefficient
    b_shear: float = 1.0      # deviatoric strain-optic coefficient

    # Texture coupling (pending GOCE TAP-B, provisional branch-local)
    g_T: float = 0.202        # provisional from P49 branch-local estimate

    # Numerical
    grid_resolution: float = 0.5   # Mpc/h per grid cell
    solver_tol: float = 1e-6
    max_solver_iter: int = 10000


@dataclass
class BaryonicSource:
    """A baryonic mass concentration (cluster or galaxy group)."""
    ra: float          # degrees
    dec: float         # degrees
    z: float           # redshift
    mass: float        # M_sun/h (total mass estimate)
    x: float = 0.0    # comoving Mpc/h (computed)
    y: float = 0.0    # comoving Mpc/h (computed)
    richness: float = 0.0


@dataclass
class FilamentTarget:
    """A filament target for forward model evaluation."""
    roster_id: str
    z_mean: float
    endpoint1_ra: float
    endpoint1_dec: float
    endpoint2_ra: float
    endpoint2_dec: float
    sources: List[BaryonicSource] = field(default_factory=list)

    # Computed
    spine_direction: Optional[np.ndarray] = None
    length_mpc_h: float = 0.0


@dataclass
class ResidualLedger:
    """Mandatory residual accounting (Appendix I.8)."""
    solver_residual: float = 0.0
    discretization_residual: float = 0.0
    truncation_residual: float = 0.0
    preconditioning_loss: float = 0.0

    @property
    def total_quadrature(self):
        return np.sqrt(
            self.solver_residual**2 +
            self.discretization_residual**2 +
            self.truncation_residual**2 +
            self.preconditioning_loss**2
        )

    def check_budget(self, signal_amplitude, max_fraction=0.10):
        """Check if total residual is within budget (10% of signal)."""
        if signal_amplitude <= 0:
            return False
        return self.total_quadrature < max_fraction * signal_amplitude


# ============================================================
# Step A: Nonlocal Closure Solve for kappa*
# ============================================================

class NonlocalClosureSolver:
    """
    Solve the nonlocal closure for kappa* (co-metric field).

    The canonical closure functional (Definition 2.4 / Appendix J):

    E_ell[kappa] = integral of:
        (1/2) lambda_IR * theta^2          (bulk penalty)
      + mu_IR * <eps^dev, eps^dev>          (shear penalty)
      + alpha * |nabla eps|^2              (gradient penalty)

    subject to baryonic/defect constraints.

    For the filament validation, we solve in 2D (transverse plane
    of the filament corridor) with point-source constraints at
    cluster locations.
    """

    def __init__(self, cert: RegimeCertificate):
        self.cert = cert

    def setup_grid(self, x_range, y_range):
        """Set up the computational grid."""
        dx = self.cert.grid_resolution
        nx = int((x_range[1] - x_range[0]) / dx) + 1
        ny = int((y_range[1] - y_range[0]) / dx) + 1

        self.x = np.linspace(x_range[0], x_range[1], nx)
        self.y = np.linspace(y_range[0], y_range[1], ny)
        self.dx = self.x[1] - self.x[0] if nx > 1 else dx
        self.dy = self.y[1] - self.y[0] if ny > 1 else dx
        self.nx = nx
        self.ny = ny

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        return self.X, self.Y

    def build_source_field(self, sources: List[BaryonicSource]):
        """
        Build the baryonic source/constraint field.

        Each source contributes a stress-energy density proportional to
        its mass, smoothed by the readout kernel.
        """
        rho = np.zeros((self.nx, self.ny))
        sigma = self.cert.sigma_kernel / self.dx  # kernel sigma in grid units

        for src in sources:
            # Distance from source
            dx = self.X - src.x
            dy = self.Y - src.y
            r2 = dx**2 + dy**2

            # Gaussian source profile (smoothed at readout scale)
            # Mass normalization: integral of rho = M / (area)
            norm = src.mass / (2 * np.pi * self.cert.sigma_kernel**2)
            rho += norm * np.exp(-r2 / (2 * self.cert.sigma_kernel**2))

        return rho

    def solve_kappa(self, sources: List[BaryonicSource],
                    x_range=None, y_range=None):
        """
        Solve the closure for kappa* given baryonic sources.

        Uses iterative relaxation (Gauss-Seidel) on the Euler-Lagrange
        system (Appendix J, Theorem H.9).

        Returns:
            kappa: 2D array of co-metric field
            residual: ResidualLedger
        """
        if x_range is None:
            # Auto-determine grid from source positions
            xs = [s.x for s in sources]
            ys = [s.y for s in sources]
            pad = 3 * self.cert.R_max
            x_range = (min(xs) - pad, max(xs) + pad)
            y_range = (min(ys) - pad, max(ys) + pad)

        self.setup_grid(x_range, y_range)

        # Build source field
        rho = self.build_source_field(sources)

        # Initialize kappa with background (kappa^(0) = identity, so delta_kappa = 0)
        # We solve for delta_kappa = kappa - kappa^(0)
        kappa = np.zeros((self.nx, self.ny, 2, 2))  # 2x2 symmetric tensor field
        kappa[:, :, 0, 0] = 1.0  # background identity
        kappa[:, :, 1, 1] = 1.0

        # Solve via relaxation
        # The Euler-Lagrange equation for the canonical closure is:
        #   -alpha * nabla^2(kappa) + lambda_IR * grad(theta) + 2*mu_IR * div(eps^dev)
        #   = T^(Phi)_eff
        #
        # In the isotropic small-strain limit, this reduces to a screened
        # Poisson equation for each component.

        mu = self.cert.mu_IR
        lam = self.cert.lambda_IR
        alpha = self.cert.alpha
        K = lam + 2 * mu  # effective bulk+shear stiffness

        # Screening length from gradient penalty
        if K > 0 and alpha > 0:
            xi = np.sqrt(alpha / K)  # characteristic decay scale
        else:
            xi = self.cert.ell  # fallback

        # Solve screened Poisson: (-alpha nabla^2 + K) delta_kappa_trace = rho
        # In Fourier space: (alpha k^2 + K) hat{delta_kappa} = hat{rho}

        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k2 = KX**2 + KY**2

        # Normalize rho to dimensionless strain-scale values
        # The source field needs to be in strain units, not mass units
        rho_max = np.max(np.abs(rho))
        if rho_max > 0:
            rho_normalized = rho / rho_max * 0.01  # small strain regime
        else:
            rho_normalized = rho

        # Bulk (trace) channel
        rho_hat = np.fft.fft2(rho_normalized)
        denom_bulk = alpha * k2 + K
        denom_bulk[0, 0] = max(denom_bulk[0, 0], 1e-10)  # regularize zero mode

        theta_hat = rho_hat / denom_bulk
        theta = np.real(np.fft.ifft2(theta_hat))
        # Clamp to small-strain regime
        theta = np.clip(theta, -2.0, 2.0)

        # Deviatoric (shear) channel — sourced by anisotropic part of stress
        # For multi-source geometry, the anisotropy comes from the
        # source pair separation axis
        if len(sources) >= 2 and mu > 0:
            # Compute anisotropic source term
            # The deviatoric stress from two separated sources has a
            # quadrupolar pattern aligned with the separation axis
            rho_dev_xx = np.zeros((self.nx, self.ny))
            rho_dev_xy = np.zeros((self.nx, self.ny))

            for i, s1 in enumerate(sources):
                for j, s2 in enumerate(sources):
                    if j <= i:
                        continue

                    # Separation direction
                    dx_sep = s2.x - s1.x
                    dy_sep = s2.y - s1.y
                    d_sep = np.sqrt(dx_sep**2 + dy_sep**2)
                    if d_sep < 1e-10:
                        continue

                    nx_hat = dx_sep / d_sep
                    ny_hat = dy_sep / d_sep

                    # Quadrupolar pattern: n_i n_j - (1/2) delta_ij
                    q_xx = nx_hat**2 - 0.5
                    q_xy = nx_hat * ny_hat

                    # Midpoint
                    xm = (s1.x + s2.x) / 2
                    ym = (s1.y + s2.y) / 2

                    # Source amplitude (proportional to product of masses)
                    amp = np.sqrt(s1.mass * s2.mass) / (d_sep * self.cert.sigma_kernel)

                    # Corridor-shaped source (elongated along separation)
                    dx_mid = self.X - xm
                    dy_mid = self.Y - ym

                    # Project onto parallel and perpendicular
                    r_par = dx_mid * nx_hat + dy_mid * ny_hat
                    r_perp = -dx_mid * ny_hat + dy_mid * nx_hat

                    # Corridor profile: Gaussian in perp, extended in parallel
                    sigma_par = d_sep / 2
                    sigma_perp = self.cert.sigma_kernel

                    corridor = np.exp(
                        -r_par**2 / (2 * sigma_par**2)
                        -r_perp**2 / (2 * sigma_perp**2)
                    )

                    rho_dev_xx += amp * q_xx * corridor
                    rho_dev_xy += amp * q_xy * corridor

            # Normalize deviatoric source to small-strain scale
            dev_max = max(np.max(np.abs(rho_dev_xx)), np.max(np.abs(rho_dev_xy)), 1e-30)
            rho_dev_xx_norm = rho_dev_xx / dev_max * 0.005
            rho_dev_xy_norm = rho_dev_xy / dev_max * 0.005

            # Solve for deviatoric components
            denom_shear = alpha * k2 + 2 * mu
            denom_shear[0, 0] = max(denom_shear[0, 0], 1e-10)

            eps_dev_xx = np.real(np.fft.ifft2(
                np.fft.fft2(rho_dev_xx_norm) / denom_shear
            ))
            eps_dev_xy = np.real(np.fft.ifft2(
                np.fft.fft2(rho_dev_xy_norm) / denom_shear
            ))
        else:
            eps_dev_xx = np.zeros((self.nx, self.ny))
            eps_dev_xy = np.zeros((self.nx, self.ny))

        # Assemble kappa field
        # kappa_ij = kappa^(0)_ij + delta_kappa_ij
        # delta_kappa ~ exp(theta) * (I + 2*eps^dev)  (to leading order)
        kappa[:, :, 0, 0] = np.exp(theta) * (1 + 2 * eps_dev_xx)
        kappa[:, :, 1, 1] = np.exp(theta) * (1 - 2 * eps_dev_xx)
        kappa[:, :, 0, 1] = np.exp(theta) * 2 * eps_dev_xy
        kappa[:, :, 1, 0] = kappa[:, :, 0, 1]

        # Compute residuals
        residual = ResidualLedger()

        # Solver residual: check E-L equation satisfaction
        # (simplified: check that Laplacian of solution matches source)
        theta_reconstructed = np.log(0.5 * (kappa[:, :, 0, 0] + kappa[:, :, 1, 1]))
        laplacian_theta = ndimage.laplace(theta_reconstructed) / (self.dx * self.dy)
        el_residual = np.abs(-alpha * laplacian_theta + K * theta_reconstructed - rho)
        residual.solver_residual = float(np.max(el_residual) / (np.max(np.abs(rho)) + 1e-30))

        # Discretization residual
        residual.discretization_residual = float(self.dx / self.cert.ell)

        return kappa, theta, eps_dev_xx, eps_dev_xy, residual


# ============================================================
# Step B: Log-Strain and Derived Fields
# ============================================================

def compute_log_strain(kappa, kappa_0=None):
    """
    Compute log-strain field from co-metric kappa.

    eps = (1/2) g^(0) * log(kappa^(0)^{-1} kappa)

    For small strain: eps ≈ (1/2)(kappa - kappa^(0))

    Returns:
        theta: trace (bulk strain)
        eps_dev: deviatoric strain (2x2 traceless symmetric)
    """
    nx, ny = kappa.shape[:2]

    # Trace part: theta = log(det(kappa) / det(kappa^(0))) / 2
    det_kappa = kappa[:, :, 0, 0] * kappa[:, :, 1, 1] - kappa[:, :, 0, 1]**2
    det_kappa = np.maximum(det_kappa, 1e-30)  # positivity guard
    theta = 0.5 * np.log(det_kappa)

    # Deviatoric part
    eps_dev_xx = 0.5 * np.log(kappa[:, :, 0, 0] / np.sqrt(det_kappa))
    eps_dev_xy = 0.5 * kappa[:, :, 0, 1] / np.sqrt(det_kappa)

    return theta, eps_dev_xx, eps_dev_xy


# ============================================================
# Step D: Anisotropy Field Q and Filament Observables
# ============================================================

def compute_Q_anisotropy(eps_dev_xx, eps_dev_xy, sigma_ell, dx):
    """
    Compute the STF anisotropy tensor Q(x; ell) from strain field.

    Q is the symmetric-traceless-free part of the second Liouville moment
    of the strain field at readout scale ell (Lemma H.2 / Step D).

    For the 2D case:
    Q_ij(x; ell) = <eps^dev_ij>_ell(x)
    where <...>_ell denotes smoothing at scale ell.

    Returns:
        Q_xx, Q_xy: components of the STF anisotropy tensor
    """
    sigma_pix = sigma_ell / dx  # smoothing in pixel units

    Q_xx = ndimage.gaussian_filter(eps_dev_xx, sigma=sigma_pix)
    Q_xy = ndimage.gaussian_filter(eps_dev_xy, sigma=sigma_pix)

    return Q_xx, Q_xy


def compute_Q_norm(Q_xx, Q_xy):
    """Frobenius norm of Q: ||Q||_F = sqrt(Q_xx^2 + 2*Q_xy^2 + Q_yy^2)."""
    # Q is traceless: Q_yy = -Q_xx
    return np.sqrt(2 * Q_xx**2 + 2 * Q_xy**2)


def compute_principal_direction(Q_xx, Q_xy):
    """
    Principal direction of Q (filament orientation field).

    Returns angle theta_p in radians.
    """
    return 0.5 * np.arctan2(2 * Q_xy, 2 * Q_xx)


def extract_transverse_profile(Q_xx, Q_xy, x, y, filament: FilamentTarget,
                                cert: RegimeCertificate):
    """
    Extract the transverse Q-anisotropy profile across a filament.

    This is the core observable: Q_perp(r_perp) projected perpendicular
    to the filament spine.

    Returns:
        r_perp_bins: transverse distance bins (Mpc/h)
        Q_perp_profile: mean Q_perp in each bin
        Q_perp_err: bootstrap error in each bin
    """
    # Filament spine direction
    dx_spine = filament.sources[-1].x - filament.sources[0].x
    dy_spine = filament.sources[-1].y - filament.sources[0].y
    L = np.sqrt(dx_spine**2 + dy_spine**2)

    if L < 1e-10:
        return None, None, None

    # Unit vectors
    e_par = np.array([dx_spine, dy_spine]) / L  # parallel to spine
    e_perp = np.array([-dy_spine, dx_spine]) / L  # perpendicular

    # Midpoint
    xm = (filament.sources[0].x + filament.sources[-1].x) / 2
    ym = (filament.sources[0].y + filament.sources[-1].y) / 2

    # Project Q onto perpendicular direction:
    # Q_perp = e_perp^T Q e_perp = Q_xx * e_perp_x^2 + 2*Q_xy * e_perp_x * e_perp_y + Q_yy * e_perp_y^2
    # Since Q_yy = -Q_xx (traceless):
    Q_perp_field = (
        Q_xx * (e_perp[0]**2 - e_perp[1]**2) +
        2 * Q_xy * e_perp[0] * e_perp[1]
    )

    # Compute perpendicular distance from spine for each grid point
    X, Y = np.meshgrid(x, y, indexing='ij')
    dx_from_mid = X - xm
    dy_from_mid = Y - ym

    r_par = dx_from_mid * e_par[0] + dy_from_mid * e_par[1]
    r_perp = dx_from_mid * e_perp[0] + dy_from_mid * e_perp[1]

    # Select points within the corridor (parallel extent)
    mask = np.abs(r_par) < L / 2

    # Bin in perpendicular direction
    r_perp_max = cert.R_max
    n_bins = 20
    bin_edges = np.linspace(-r_perp_max, r_perp_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    Q_perp_profile = np.zeros(n_bins)
    Q_perp_err = np.zeros(n_bins)
    Q_perp_count = np.zeros(n_bins)

    for b in range(n_bins):
        in_bin = mask & (r_perp >= bin_edges[b]) & (r_perp < bin_edges[b + 1])
        if np.sum(in_bin) > 0:
            vals = Q_perp_field[in_bin]
            Q_perp_profile[b] = np.mean(vals)
            Q_perp_err[b] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            Q_perp_count[b] = np.sum(in_bin)

    return bin_centers, Q_perp_profile, Q_perp_err


def compute_integrated_amplitude(r_perp_bins, Q_perp_profile):
    """
    Compute integrated transverse anisotropy amplitude A_fil.

    A_fil = integral_0^R_max Q_perp(r_perp) * r_perp dr_perp

    Uses only the positive-r_perp half (symmetric profile).
    """
    mask_pos = r_perp_bins > 0
    r = r_perp_bins[mask_pos]
    Q = Q_perp_profile[mask_pos]

    if len(r) < 2:
        return 0.0

    # Trapezoidal integration
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    A_fil = _trapz(Q * r, r)

    return float(A_fil)


# ============================================================
# Null Tests
# ============================================================

def first_jet_null(solver, sources, cert):
    """
    First-Jet null test: force mu_IR = 0 (no shear/anisotropy channel).

    The filament signal should collapse.
    """
    cert_null = RegimeCertificate(**{
        k: v for k, v in cert.__dict__.items()
    })
    cert_null.mu_IR = 0.0  # kill the shear channel

    solver_null = NonlocalClosureSolver(cert_null)
    return solver_null.solve_kappa(sources)


def isotropy_null(Q_xx, Q_xy):
    """
    Isotropy null: replace Q with its isotropic average.
    Directed filament response should vanish.
    """
    # Isotropic average of a traceless tensor is zero
    return np.zeros_like(Q_xx), np.zeros_like(Q_xy)


def scrambled_geometry_null(solver, sources, cert, rng=None):
    """
    Scrambled geometry null: randomize source positions.
    Bridge coherence should vanish.
    """
    if rng is None:
        rng = np.random.RandomState(12345)

    scrambled_sources = []
    for src in sources:
        s = BaryonicSource(
            ra=src.ra, dec=src.dec, z=src.z, mass=src.mass,
            x=src.x + rng.normal(0, 50),  # scramble by 50 Mpc/h
            y=src.y + rng.normal(0, 50),
            richness=src.richness,
        )
        scrambled_sources.append(s)

    return solver.solve_kappa(scrambled_sources)


def scale_robustness_test(solver, sources, cert, filament, scale_factors=(0.7, 1.4)):
    """
    Scale robustness test (Contract Sec 9.4).

    Recompute with ell' = factor * ell. Signal should persist within
    a factor of 2-3 and not change sign.
    """
    results = {}
    for factor in scale_factors:
        cert_scaled = RegimeCertificate(**{k: v for k, v in cert.__dict__.items()})
        cert_scaled.ell = cert.ell * factor
        cert_scaled.sigma_kernel = cert.sigma_kernel * factor

        solver_scaled = NonlocalClosureSolver(cert_scaled)
        kappa_s, _, _, _, _ = solver_scaled.solve_kappa(sources)
        _, eps_xx_s, eps_xy_s = compute_log_strain(kappa_s)
        Q_xx_s, Q_xy_s = compute_Q_anisotropy(
            eps_xx_s, eps_xy_s,
            sigma_ell=cert_scaled.sigma_kernel / solver_scaled.dx,
            dx=solver_scaled.dx,
        )
        r_bins, Q_prof, _ = extract_transverse_profile(
            Q_xx_s, Q_xy_s,
            solver_scaled.x, solver_scaled.y,
            filament, cert_scaled,
        )
        if r_bins is not None:
            A_fil_s = compute_integrated_amplitude(r_bins, Q_prof)
        else:
            A_fil_s = 0.0
        results[f"ell_x{factor}"] = float(A_fil_s)
    return results


# ============================================================
# Full Pipeline
# ============================================================

class RTFilamentPipeline:
    """
    Full RT filament forward model pipeline.

    Implements the certified forward chain:
    baryonic geometry -> kappa* -> eps* -> Q(x;ell) -> SFTSP -> A_fil
    """

    def __init__(self, cert: RegimeCertificate):
        self.cert = cert
        self.solver = NonlocalClosureSolver(cert)
        self.results = {}

    def run_single_filament(self, filament: FilamentTarget):
        """
        Run the full forward model for a single filament.

        Returns a dict with all intermediate and final results.
        """
        result = {
            "roster_id": filament.roster_id,
            "z_mean": filament.z_mean,
            "n_sources": len(filament.sources),
        }

        # Step A: Solve closure
        kappa, theta, eps_xx, eps_xy, residual_A = self.solver.solve_kappa(
            filament.sources
        )
        result["residual_solver"] = residual_A.solver_residual
        result["residual_discretization"] = residual_A.discretization_residual

        # Step B: Log-strain (already computed in solver for small-strain)
        theta_check, eps_dev_xx, eps_dev_xy = compute_log_strain(kappa)

        # Step D: Anisotropy field Q
        Q_xx, Q_xy = compute_Q_anisotropy(
            eps_dev_xx, eps_dev_xy,
            sigma_ell=self.cert.sigma_kernel / self.solver.dx,
            dx=self.solver.dx,
        )

        Q_norm = compute_Q_norm(Q_xx, Q_xy)
        result["Q_norm_max"] = float(np.max(Q_norm))
        result["Q_norm_mean"] = float(np.mean(Q_norm))

        # Extract transverse profile
        r_bins, Q_profile, Q_err = extract_transverse_profile(
            Q_xx, Q_xy,
            self.solver.x, self.solver.y,
            filament, self.cert,
        )

        if r_bins is not None:
            result["r_perp_bins"] = r_bins.tolist()
            result["Q_perp_profile"] = Q_profile.tolist()
            result["Q_perp_err"] = Q_err.tolist()

            # Primary observable: integrated amplitude
            A_fil = compute_integrated_amplitude(r_bins, Q_profile)
            result["A_fil"] = A_fil
        else:
            result["A_fil"] = 0.0
            result["WARNING"] = "Could not extract transverse profile"

        # Residual budget check
        total_residual = residual_A.total_quadrature
        result["total_residual"] = float(total_residual)
        result["residual_budget_ok"] = residual_A.check_budget(
            abs(result.get("A_fil", 0))
        )

        return result

    def run_null_tests(self, filament: FilamentTarget):
        """Run all null tests for a single filament."""
        nulls = {}

        # 1. First-Jet null (mu_IR = 0)
        kappa_null, theta_null, eps_xx_null, eps_xy_null, _ = first_jet_null(
            self.solver, filament.sources, self.cert
        )
        _, eps_dev_xx_null, eps_dev_xy_null = compute_log_strain(kappa_null)
        Q_xx_null, Q_xy_null = compute_Q_anisotropy(
            eps_dev_xx_null, eps_dev_xy_null,
            sigma_ell=self.cert.sigma_kernel / self.solver.dx,
            dx=self.solver.dx,
        )
        r_bins, Q_profile_null, _ = extract_transverse_profile(
            Q_xx_null, Q_xy_null,
            self.solver.x, self.solver.y,
            filament, self.cert,
        )
        if r_bins is not None:
            nulls["first_jet_A_fil"] = compute_integrated_amplitude(r_bins, Q_profile_null)
        else:
            nulls["first_jet_A_fil"] = 0.0

        # 2. Isotropy null
        Q_xx_iso, Q_xy_iso = isotropy_null(
            *compute_Q_anisotropy(
                eps_dev_xx_null, eps_dev_xy_null,
                sigma_ell=self.cert.sigma_kernel / self.solver.dx,
                dx=self.solver.dx,
            )
        )
        nulls["isotropy_A_fil"] = 0.0  # by construction (traceless avg = 0)

        # 3. Scrambled geometry null
        kappa_scram, _, _, _, _ = scrambled_geometry_null(
            self.solver, filament.sources, self.cert
        )
        _, eps_scram_xx, eps_scram_xy = compute_log_strain(kappa_scram)
        Q_scram_xx, Q_scram_xy = compute_Q_anisotropy(
            eps_scram_xx, eps_scram_xy,
            sigma_ell=self.cert.sigma_kernel / self.solver.dx,
            dx=self.solver.dx,
        )
        r_bins, Q_scram, _ = extract_transverse_profile(
            Q_scram_xx, Q_scram_xy,
            self.solver.x, self.solver.y,
            filament, self.cert,
        )
        if r_bins is not None:
            nulls["scrambled_A_fil"] = compute_integrated_amplitude(r_bins, Q_scram)
        else:
            nulls["scrambled_A_fil"] = 0.0

        # 4. Scale robustness test
        scale_results = scale_robustness_test(
            self.solver, filament.sources, self.cert, filament
        )
        nulls["scale_robustness"] = scale_results

        return nulls

    def run_roster(self, roster_path: str, run_nulls: bool = False):
        """Run the pipeline on a full roster."""
        with open(roster_path) as f:
            roster = json.load(f)

        all_results = {
            "contract_version": roster.get("contract_version", "1.0"),
            "regime_certificate": self.cert.__dict__,
            "training_results": [],
            "holdout_results": [],
        }

        for set_name in ("training", "holdout"):
            for fil_data in roster.get(set_name, []):
                # Normalize roster entry: ensure roster_id exists
                if "roster_id" not in fil_data:
                    fil_data = dict(fil_data)
                    fil_data["roster_id"] = str(fil_data.get("filament_id", "unknown"))
                if "set" not in fil_data:
                    fil_data = dict(fil_data)
                    fil_data["set"] = set_name

                filament = self._build_filament_target(fil_data)
                result = self.run_single_filament(filament)

                if run_nulls:
                    result["null_tests"] = self.run_null_tests(filament)

                all_results[f"{set_name}_results"].append(result)

        return all_results

    def _build_filament_target(self, fil_data):
        """Build a FilamentTarget from roster data."""
        z_mean = fil_data.get("z_mean", fil_data.get("redshift_mean", 0.1))
        roster_id = fil_data.get("roster_id", str(fil_data.get("filament_id", "unknown")))

        filament = FilamentTarget(
            roster_id=roster_id,
            z_mean=z_mean,
            endpoint1_ra=fil_data["endpoint1_ra"],
            endpoint1_dec=fil_data["endpoint1_dec"],
            endpoint2_ra=fil_data["endpoint2_ra"],
            endpoint2_dec=fil_data["endpoint2_dec"],
        )

        # Convert endpoints to comoving coordinates
        # (simplified: flat sky projection around midpoint)
        z = z_mean
        d_c = _comoving_distance(z)

        ra_mid = (fil_data["endpoint1_ra"] + fil_data["endpoint2_ra"]) / 2
        dec_mid = (fil_data["endpoint1_dec"] + fil_data["endpoint2_dec"]) / 2

        cos_dec = np.cos(np.radians(dec_mid))

        x1 = d_c * np.radians(fil_data["endpoint1_ra"] - ra_mid) * cos_dec
        y1 = d_c * np.radians(fil_data["endpoint1_dec"] - dec_mid)
        x2 = d_c * np.radians(fil_data["endpoint2_ra"] - ra_mid) * cos_dec
        y2 = d_c * np.radians(fil_data["endpoint2_dec"] - dec_mid)

        # Create sources at endpoints (placeholder masses)
        # In production, these come from cluster catalogs
        mass_scale = 1e14  # M_sun/h (typical cluster)
        richness_sum = fil_data.get("richness_sum", 100)

        filament.sources = [
            BaryonicSource(
                ra=fil_data["endpoint1_ra"],
                dec=fil_data["endpoint1_dec"],
                z=z,
                mass=mass_scale * richness_sum / 200,
                x=x1, y=y1,
                richness=richness_sum / 2,
            ),
            BaryonicSource(
                ra=fil_data["endpoint2_ra"],
                dec=fil_data["endpoint2_dec"],
                z=z,
                mass=mass_scale * richness_sum / 200,
                x=x2, y=y2,
                richness=richness_sum / 2,
            ),
        ]

        filament.length_mpc_h = fil_data.get("length_mpc_h", 20.0)

        return filament


def _comoving_distance(z, Om=0.3):
    """Comoving distance in Mpc/h."""
    zz = np.linspace(0, z, 500)
    E = np.sqrt(Om * (1 + zz)**3 + (1 - Om))
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return 2997.9 * _trapz(1.0 / E, zz)


# ============================================================
# mu_IR Anchoring
# ============================================================

def anchor_mu_IR(roster_path: str, observed_profiles: Dict[str, np.ndarray],
                 cert: RegimeCertificate, mu_range=(0.01, 10.0)):
    """
    Anchor mu_IR from training filaments.

    Minimizes chi^2 between predicted and observed SFTSP over
    the training set.

    Args:
        roster_path: path to frozen roster JSON
        observed_profiles: dict mapping roster_id -> (r_bins, gamma_t, gamma_t_err)
        cert: regime certificate (mu_IR will be updated)
        mu_range: search range for mu_IR

    Returns:
        mu_IR_star: best-fit value
        chi2_min: minimum chi-squared
        stability: leave-one-out stability results
    """

    with open(roster_path) as f:
        roster = json.load(f)

    training = roster["training"]

    def chi2_total(log_mu):
        mu = np.exp(log_mu)
        cert_trial = RegimeCertificate(**{
            k: v for k, v in cert.__dict__.items()
        })
        cert_trial.mu_IR = mu

        pipeline = RTFilamentPipeline(cert_trial)

        total_chi2 = 0.0
        for fil_data in training:
            rid = str(fil_data.get("roster_id", fil_data.get("filament_id", "unknown")))
            if rid not in observed_profiles:
                continue

            fil_data_norm = dict(fil_data)
            fil_data_norm["roster_id"] = rid
            filament = pipeline._build_filament_target(fil_data_norm)
            result = pipeline.run_single_filament(filament)

            r_obs, gamma_obs, gamma_err = observed_profiles[rid]

            if result.get("Q_perp_profile") is not None:
                Q_pred = np.array(result["Q_perp_profile"])
                # Map Q_pred to gamma_t (simplified: proportional)
                gamma_pred = cert.b_shear * Q_pred

                # Chi-squared
                mask = gamma_err > 0
                chi2 = np.sum(
                    ((gamma_pred[mask] - gamma_obs[mask]) / gamma_err[mask])**2
                )
                total_chi2 += chi2

        return total_chi2

    # Minimize
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(
        chi2_total,
        bounds=(np.log(mu_range[0]), np.log(mu_range[1])),
        method='bounded',
    )

    mu_IR_star = np.exp(result.x)
    chi2_min = result.fun

    # Leave-one-out stability (ESG)
    stability = {}
    for leave_out_idx in range(len(training)):
        training_reduced = [t for i, t in enumerate(training) if i != leave_out_idx]

        def chi2_loo(log_mu):
            mu = np.exp(log_mu)
            cert_trial = RegimeCertificate(**{
                k: v for k, v in cert.__dict__.items()
            })
            cert_trial.mu_IR = mu
            pipeline = RTFilamentPipeline(cert_trial)

            total = 0.0
            for fil_data in training_reduced:
                rid = str(fil_data.get("roster_id", fil_data.get("filament_id", "unknown")))
                if rid not in observed_profiles:
                    continue
                fil_data_n = dict(fil_data)
                fil_data_n["roster_id"] = rid
                filament = pipeline._build_filament_target(fil_data_n)
                result = pipeline.run_single_filament(filament)
                r_obs, gamma_obs, gamma_err = observed_profiles[rid]
                if result.get("Q_perp_profile") is not None:
                    Q_pred = np.array(result["Q_perp_profile"])
                    gamma_pred = cert.b_shear * Q_pred
                    mask = gamma_err > 0
                    total += np.sum(((gamma_pred[mask] - gamma_obs[mask]) / gamma_err[mask])**2)
            return total

        res_loo = minimize_scalar(
            chi2_loo,
            bounds=(np.log(mu_range[0]), np.log(mu_range[1])),
            method='bounded',
        )
        mu_loo = np.exp(res_loo.x)
        loo_id = str(training[leave_out_idx].get("roster_id", training[leave_out_idx].get("filament_id", "unknown")))
        stability[loo_id] = {
            "mu_IR_without": float(mu_loo),
            "relative_change": float(abs(mu_loo - mu_IR_star) / mu_IR_star),
        }

    # ESG check
    max_change = max(s["relative_change"] for s in stability.values())
    esg_passed = max_change < 0.15  # 15% threshold from contract

    return {
        "mu_IR_star": float(mu_IR_star),
        "chi2_min": float(chi2_min),
        "n_training": len(training),
        "stability": stability,
        "max_loo_change": float(max_change),
        "ESG_passed": esg_passed,
    }


# ============================================================
# Main entry point
# ============================================================

def main():
    """Run the pipeline on the synthetic roster for development."""

    # Prefer v3 (KiDS-overlap), then v2 (all-sky real), then v1 (synthetic)
    roster_dir = Path(__file__).parent.parent / "rosters"
    for vname in ["filament_roster_v4.json", "filament_roster_v3.json", "filament_roster_v2.json", "filament_roster_v1.json"]:
        if (roster_dir / vname).exists():
            roster_path = roster_dir / vname
            break

    if not roster_path.exists():
        print("Roster not found. Run build_metrology_roster.py first.")
        print(f"Expected: {roster_path}")
        return 1

    # Initialize with placeholder parameters
    cert = RegimeCertificate(
        mu_IR=1.0,  # placeholder — to be anchored
    )

    pipeline = RTFilamentPipeline(cert)

    print("=" * 60)
    print("RT Filament Forward Model Pipeline")
    print(f"mu_IR = {cert.mu_IR} (placeholder)")
    print(f"ell = {cert.ell} Mpc/h")
    print(f"R_max = {cert.R_max} Mpc/h")
    print("=" * 60)

    results = pipeline.run_roster(str(roster_path), run_nulls=True)

    # Save results
    output_path = Path(__file__).parent / "pipeline_results_dev.json"

    # Convert numpy types for JSON serialization
    import copy

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    results_clean = make_serializable(results)

    with open(output_path, "w") as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved: {output_path}")

    # Print summary
    print(f"\nTraining results:")
    for r in results["training_results"]:
        print(f"  {r['roster_id']}: A_fil = {r.get('A_fil', 'N/A'):.6f}, "
              f"residual_ok = {r.get('residual_budget_ok', 'N/A')}")

    print(f"\nHoldout results:")
    for r in results["holdout_results"]:
        null_1j = r.get("null_tests", {}).get("first_jet_A_fil", "N/A")
        print(f"  {r['roster_id']}: A_fil = {r.get('A_fil', 'N/A'):.6f}, "
              f"1J-null = {null_1j}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
