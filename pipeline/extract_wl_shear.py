#!/usr/bin/env python3
"""
Extract weak-lensing tangential shear profiles from KiDS-1000 gold catalog
around each filament in the frozen roster.

This module provides the observed gamma_t(r_perp) profiles needed for
mu_IR anchoring (training set) and holdout evaluation.

Method:
1. Load KiDS-1000 gold WL catalog (RA, Dec, e1, e2, weight, z_B)
2. For each filament in the roster:
   a. Select background sources (z_source > z_filament + delta_z_buffer)
   b. Define filament spine midpoint and orientation
   c. Select sources within R_max corridor
   d. Compute tangential shear relative to the spine direction
   e. Stack in transverse bins
3. Output: {roster_id -> (r_perp_bins, gamma_t, gamma_t_err)}

Reference:
- Giblin+ 2021, A&A 645, A105 (KiDS-1000 shape measurements)
- Hildebrandt+ 2021, A&A 647, A124 (KiDS-1000 photo-z)
- de Graaff+ 2019, A&A 624, A48 (filament lensing methodology)
"""

import json
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

try:
    from astropy.io import fits
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


# ============================================================
# Configuration
# ============================================================

@dataclass
class ShearExtractionConfig:
    """Configuration for WL shear extraction around filaments."""

    # Source selection
    delta_z_buffer: float = 0.1    # z_source > z_fil + delta_z for background
    z_source_max: float = 1.2      # max source redshift (quality cut)

    # Corridor geometry
    R_max_mpc_h: float = 5.0       # max transverse distance (Mpc/h)
    L_buffer_mpc_h: float = 2.0    # extend corridor beyond endpoints

    # Binning
    n_bins: int = 20               # number of transverse bins
    symmetric: bool = True         # fold profile about r_perp=0

    # Quality cuts
    weight_min: float = 0.0        # minimum lensfit weight
    fitclass_values: tuple = (0,)  # accepted fitclass values (0 = good star/gal separation)

    # Cosmology (for angular-to-physical conversion)
    Om0: float = 0.3               # matter density
    h: float = 0.7                 # Hubble parameter H0/(100 km/s/Mpc)


# ============================================================
# Utility: comoving distance and angular diameter distance
# ============================================================

def comoving_distance(z, Om=0.3):
    """Comoving distance in Mpc/h."""
    zz = np.linspace(0, z, 500)
    E = np.sqrt(Om * (1 + zz)**3 + (1 - Om))
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return 2997.9 * _trapz(1.0 / E, zz)


def angular_diameter_distance(z, Om=0.3):
    """Angular diameter distance in Mpc/h."""
    return comoving_distance(z, Om) / (1 + z)


def sigma_crit_inv(z_l, z_s, Om=0.3):
    """
    Inverse critical surface density (geometric lensing efficiency).

    Sigma_crit^{-1} = (4 pi G / c^2) * D_l * D_ls / D_s

    Returns in units of h Mpc^{-2} M_sun^{-1} (natural for our pipeline).
    For the ratio gamma_t / kappa this cancels, so we mainly need
    the relative weighting across source redshifts.
    """
    if z_s <= z_l:
        return 0.0

    d_l = angular_diameter_distance(z_l, Om)
    d_s = angular_diameter_distance(z_s, Om)
    d_ls = comoving_distance(z_s, Om) / (1 + z_s) - comoving_distance(z_l, Om) / (1 + z_s)
    # d_ls can be approximated for flat LCDM
    if d_ls <= 0 or d_s <= 0:
        return 0.0

    # 4piG/c^2 in Mpc/M_sun * h units: 6.013e-19 Mpc/M_sun
    four_pi_G_c2 = 6.013e-19  # Mpc h^{-1} M_sun^{-1} h
    return four_pi_G_c2 * d_l * d_ls / d_s


# ============================================================
# KiDS catalog loading
# ============================================================

def load_kids_catalog(catalog_path, config=None):
    """
    Load KiDS-1000 gold WL catalog.

    Expected columns (KiDS-1000 gold catalog):
    - ALPHA_J2000 (or RAJ2000): RA in degrees
    - DELTA_J2000 (or DECJ2000): Dec in degrees
    - e1, e2: ellipticity components (or e1_A, e2_A for lensfit)
    - weight: lensfit weight
    - Z_B: best-fit photometric redshift (BPZ)
    - FITCLASS or CLASS_STAR: star/galaxy classification

    Returns:
        dict with keys: ra, dec, e1, e2, weight, z_B (all numpy arrays)
    """
    if not HAS_ASTROPY:
        raise ImportError("astropy required: pip install astropy")

    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"KiDS catalog not found: {catalog_path}")

    print(f"Loading KiDS catalog: {catalog_path}")
    print(f"  File size: {catalog_path.stat().st_size / 1e9:.2f} GB")

    # Read FITS — for large files, use memmap
    hdul = fits.open(str(catalog_path), memmap=True)

    # Inspect available extensions and columns
    print(f"  HDU list: {[h.name for h in hdul]}")

    # The gold catalog is typically in extension 1
    data = hdul[1].data
    cols = [c.name for c in hdul[1].columns]
    print(f"  Columns ({len(cols)}): {cols[:20]}...")
    print(f"  Number of sources: {len(data):,}")

    # Flexible column name mapping
    col_map = {}

    # RA
    for name in ['ALPHA_J2000', 'RAJ2000', 'RA', 'ra']:
        if name in cols:
            col_map['ra'] = name
            break

    # Dec
    for name in ['DELTA_J2000', 'DECJ2000', 'DEC', 'dec']:
        if name in cols:
            col_map['dec'] = name
            break

    # Ellipticity
    for name in ['e1', 'e1_A', 'E1']:
        if name in cols:
            col_map['e1'] = name
            break
    for name in ['e2', 'e2_A', 'E2']:
        if name in cols:
            col_map['e2'] = name
            break

    # Weight
    for name in ['weight', 'WEIGHT', 'w']:
        if name in cols:
            col_map['weight'] = name
            break

    # Photo-z
    for name in ['Z_B', 'z_B', 'PHOTOZ', 'Z_PHOT', 'zphot']:
        if name in cols:
            col_map['z_B'] = name
            break

    print(f"  Column mapping: {col_map}")

    required = ['ra', 'dec', 'e1', 'e2', 'weight', 'z_B']
    missing = [k for k in required if k not in col_map]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {cols}"
        )

    # Extract arrays
    catalog = {}
    for key, col_name in col_map.items():
        catalog[key] = np.array(data[col_name], dtype=np.float64)

    # Apply quality cuts
    if config is None:
        config = ShearExtractionConfig()

    mask = catalog['weight'] > config.weight_min
    mask &= catalog['z_B'] < config.z_source_max
    mask &= np.isfinite(catalog['e1'])
    mask &= np.isfinite(catalog['e2'])

    # Fitclass cut if available
    for name in ['FITCLASS', 'fitclass', 'CLASS']:
        if name in cols:
            fitclass = np.array(data[name])
            mask &= np.isin(fitclass, config.fitclass_values)
            print(f"  Applied fitclass cut: {config.fitclass_values}")
            break

    n_before = len(catalog['ra'])
    for key in catalog:
        catalog[key] = catalog[key][mask]
    n_after = len(catalog['ra'])
    print(f"  Quality cuts: {n_before:,} -> {n_after:,} sources ({100*n_after/n_before:.1f}%)")

    hdul.close()
    return catalog


# ============================================================
# Tangential shear computation
# ============================================================

def compute_tangential_shear(e1, e2, phi):
    """
    Compute tangential and cross-component shear.

    gamma_t = -(e1 * cos(2*phi) + e2 * sin(2*phi))
    gamma_x = +(e1 * sin(2*phi) - e2 * cos(2*phi))

    where phi is the position angle from the reference direction
    (perpendicular to filament spine).
    """
    cos2phi = np.cos(2 * phi)
    sin2phi = np.sin(2 * phi)

    gamma_t = -(e1 * cos2phi + e2 * sin2phi)
    gamma_x = e1 * sin2phi - e2 * cos2phi

    return gamma_t, gamma_x


def extract_filament_shear_profile(
    catalog, filament_data, config=None
):
    """
    Extract the tangential shear profile around a single filament.

    Args:
        catalog: dict from load_kids_catalog
        filament_data: dict with filament roster entry
        config: ShearExtractionConfig

    Returns:
        r_perp_bins: bin centers in Mpc/h
        gamma_t: mean tangential shear in each bin
        gamma_t_err: error on mean (shape noise + Poisson)
        gamma_x: cross-component (null diagnostic)
        n_sources: number of sources per bin
    """
    if config is None:
        config = ShearExtractionConfig()

    z_fil = filament_data.get("redshift_mean", filament_data.get("z_mean", 0.15))

    # Filament spine geometry
    ra1, dec1 = filament_data["endpoint1_ra"], filament_data["endpoint1_dec"]
    ra2, dec2 = filament_data["endpoint2_ra"], filament_data["endpoint2_dec"]

    ra_mid = (ra1 + ra2) / 2
    dec_mid = (dec1 + dec2) / 2

    # Angular diameter distance at filament redshift
    d_A = angular_diameter_distance(z_fil, config.Om0)

    # Convert R_max to angular scale
    R_max_deg = np.degrees(config.R_max_mpc_h / d_A)
    cos_dec_mid = np.cos(np.radians(dec_mid))

    # Spine direction in angular coordinates
    dra = (ra2 - ra1) * cos_dec_mid
    ddec = dec2 - dec1
    spine_len_deg = np.sqrt(dra**2 + ddec**2)
    if spine_len_deg < 1e-6:
        return None, None, None, None, None

    e_par = np.array([dra, ddec]) / spine_len_deg   # unit parallel
    e_perp = np.array([-ddec, dra]) / spine_len_deg  # unit perpendicular

    # Pre-filter sources: broad box cut for speed
    L_half_deg = spine_len_deg / 2 + np.degrees(config.L_buffer_mpc_h / d_A)
    box_half = max(L_half_deg, R_max_deg) + 0.5  # generous padding

    mask_box = (
        (np.abs(catalog['ra'] - ra_mid) * cos_dec_mid < box_half) &
        (np.abs(catalog['dec'] - dec_mid) < box_half)
    )

    # Background source selection: z_source > z_fil + buffer
    mask_bg = catalog['z_B'] > (z_fil + config.delta_z_buffer)

    mask = mask_box & mask_bg
    if np.sum(mask) < 10:
        print(f"  WARNING: Only {np.sum(mask)} background sources near filament "
              f"{filament_data.get('filament_id', '?')}")
        return None, None, None, None, None

    # Extract selected sources
    ra_s = catalog['ra'][mask]
    dec_s = catalog['dec'][mask]
    e1_s = catalog['e1'][mask]
    e2_s = catalog['e2'][mask]
    w_s = catalog['weight'][mask]
    z_s = catalog['z_B'][mask]

    # Project source positions relative to filament midpoint
    dx_deg = (ra_s - ra_mid) * cos_dec_mid
    dy_deg = dec_s - dec_mid

    # Parallel and perpendicular distances (in degrees)
    r_par_deg = dx_deg * e_par[0] + dy_deg * e_par[1]
    r_perp_deg = dx_deg * e_perp[0] + dy_deg * e_perp[1]

    # Convert to physical Mpc/h
    r_perp_mpc = r_perp_deg * np.radians(1) * d_A / np.radians(1)
    # Correct: r_perp_mpc = tan(r_perp_deg * pi/180) * d_A ≈ r_perp_deg * pi/180 * d_A
    r_perp_mpc = np.radians(np.abs(r_perp_deg)) * d_A
    r_par_mpc = np.radians(np.abs(r_par_deg)) * d_A

    # Signed perpendicular distance
    r_perp_signed = np.sign(r_perp_deg) * r_perp_mpc

    # Corridor selection
    L_half_mpc = filament_data.get("length_mpc_h", 20.0) / 2 + config.L_buffer_mpc_h
    mask_corridor = (r_par_mpc < L_half_mpc) & (r_perp_mpc < config.R_max_mpc_h)

    if np.sum(mask_corridor) < 10:
        print(f"  WARNING: Only {np.sum(mask_corridor)} sources in corridor for filament "
              f"{filament_data.get('filament_id', '?')}")
        return None, None, None, None, None

    # Position angle of each source relative to filament perpendicular
    # phi = angle between source position vector and reference direction
    # For filament lensing, the tangential direction is perpendicular to
    # the line connecting the source to the nearest spine point
    phi = np.arctan2(dy_deg[mask_corridor], dx_deg[mask_corridor])

    # Spine position angle
    phi_spine = np.arctan2(e_par[1], e_par[0])

    # Tangential shear relative to perpendicular direction
    # phi_t = phi - phi_spine - pi/2 (perpendicular to spine)
    phi_t = phi - phi_spine

    gamma_t, gamma_x = compute_tangential_shear(
        e1_s[mask_corridor], e2_s[mask_corridor], phi_t
    )
    w_corr = w_s[mask_corridor]
    r_perp_sel = np.abs(r_perp_signed[mask_corridor])

    # Lensing efficiency weighting
    sigma_crit_weights = np.array([
        sigma_crit_inv(z_fil, zs, config.Om0) for zs in z_s[mask_corridor]
    ])
    # Combined weight: shape weight * lensing efficiency
    w_total = w_corr * sigma_crit_weights

    # Bin in |r_perp|
    bin_edges = np.linspace(0, config.R_max_mpc_h, config.n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    gamma_t_binned = np.zeros(config.n_bins)
    gamma_x_binned = np.zeros(config.n_bins)
    gamma_t_err = np.zeros(config.n_bins)
    n_per_bin = np.zeros(config.n_bins, dtype=int)

    for b in range(config.n_bins):
        in_bin = (r_perp_sel >= bin_edges[b]) & (r_perp_sel < bin_edges[b + 1])
        n_in = np.sum(in_bin)
        n_per_bin[b] = n_in

        if n_in < 2:
            continue

        w_bin = w_total[in_bin]
        w_sum = np.sum(w_bin)
        if w_sum <= 0:
            continue

        gamma_t_binned[b] = np.sum(w_bin * gamma_t[in_bin]) / w_sum
        gamma_x_binned[b] = np.sum(w_bin * gamma_x[in_bin]) / w_sum

        # Error: shape noise / sqrt(N_eff)
        # Shape noise sigma_e ~ 0.28 per component (KiDS-1000)
        sigma_e = 0.28
        # Effective number of sources
        N_eff = w_sum**2 / np.sum(w_bin**2) if np.sum(w_bin**2) > 0 else n_in
        gamma_t_err[b] = sigma_e / np.sqrt(N_eff) if N_eff > 0 else 0

    return bin_centers, gamma_t_binned, gamma_t_err, gamma_x_binned, n_per_bin


# ============================================================
# Full roster extraction
# ============================================================

def extract_all_profiles(
    catalog_path, roster_path, config=None, output_path=None
):
    """
    Extract tangential shear profiles for all filaments in the roster.

    Returns:
        profiles: dict mapping roster_id -> (r_bins, gamma_t, gamma_t_err)
        diagnostics: dict with per-filament diagnostics
    """
    if config is None:
        config = ShearExtractionConfig()

    # Load catalog
    catalog = load_kids_catalog(catalog_path, config)

    # Load roster
    with open(roster_path) as f:
        roster = json.load(f)

    print(f"\nRoster: {roster.get('version', '?')}")
    print(f"  Training: {len(roster.get('training', []))} filaments")
    print(f"  Holdout: {len(roster.get('holdout', []))} filaments")

    profiles = {}
    diagnostics = {}

    for set_name in ("training", "holdout"):
        for fil_data in roster.get(set_name, []):
            fid = str(fil_data.get("filament_id", "unknown"))
            print(f"\n--- {set_name.upper()} filament {fid} ---")
            print(f"  z={fil_data.get('redshift_mean', '?'):.4f}, "
                  f"RA=[{fil_data['endpoint1_ra']:.2f}, {fil_data['endpoint2_ra']:.2f}], "
                  f"Dec=[{fil_data['endpoint1_dec']:.2f}, {fil_data['endpoint2_dec']:.2f}]")

            r_bins, gamma_t, gamma_t_err, gamma_x, n_src = extract_filament_shear_profile(
                catalog, fil_data, config
            )

            if r_bins is None:
                print(f"  FAILED: insufficient sources")
                diagnostics[fid] = {"status": "FAILED", "set": set_name}
                continue

            profiles[fid] = (r_bins, gamma_t, gamma_t_err)

            total_sources = int(np.sum(n_src))
            mean_gamma_t = float(np.mean(gamma_t[gamma_t != 0]))
            mean_gamma_x = float(np.mean(gamma_x[gamma_x != 0])) if np.any(gamma_x != 0) else 0.0

            print(f"  Sources in corridor: {total_sources:,}")
            print(f"  Mean gamma_t: {mean_gamma_t:.6f}")
            print(f"  Mean gamma_x: {mean_gamma_x:.6f} (should be ~0)")
            print(f"  S/N estimate: {abs(mean_gamma_t) / (np.mean(gamma_t_err[gamma_t_err > 0]) + 1e-10):.1f}")

            diagnostics[fid] = {
                "status": "OK",
                "set": set_name,
                "n_sources_total": total_sources,
                "n_sources_per_bin": n_src.tolist(),
                "mean_gamma_t": mean_gamma_t,
                "mean_gamma_x": mean_gamma_x,
                "gamma_t_err_median": float(np.median(gamma_t_err[gamma_t_err > 0])),
            }

    # Save results
    if output_path is None:
        output_path = Path(roster_path).parent.parent / "pipeline" / "observed_shear_profiles.json"

    output_data = {
        "catalog_file": str(catalog_path),
        "roster_file": str(roster_path),
        "config": {
            "delta_z_buffer": config.delta_z_buffer,
            "R_max_mpc_h": config.R_max_mpc_h,
            "n_bins": config.n_bins,
        },
        "profiles": {},
        "diagnostics": diagnostics,
    }

    for fid, (r_bins, gamma_t, gamma_t_err) in profiles.items():
        output_data["profiles"][fid] = {
            "r_perp_bins": r_bins.tolist(),
            "gamma_t": gamma_t.tolist(),
            "gamma_t_err": gamma_t_err.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved: {output_path}")

    return profiles, diagnostics


# ============================================================
# Main
# ============================================================

def main():
    """Extract WL shear profiles for the frozen roster."""

    # Paths
    base = Path(__file__).parent.parent
    roster_path = base / "rosters" / "filament_roster_v3.json"
    catalog_path = base / "data" / "kids_lensing" / "KiDS_DR4.1_WL_gold_cat.fits"

    if not roster_path.exists():
        print(f"Roster not found: {roster_path}")
        return 1

    if not catalog_path.exists():
        size = catalog_path.stat().st_size if catalog_path.exists() else 0
        print(f"KiDS catalog not found: {catalog_path}")
        print(f"Run download_kids_gold.py first.")
        print(f"\nTo download manually:")
        print(f"  wget -c 'https://kids.strw.leidenuniv.nl/DR4/data_files/"
              f"KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits' \\")
        print(f"    -O '{catalog_path}'")
        return 1

    print("=" * 60)
    print("WL Tangential Shear Profile Extraction")
    print(f"Catalog: {catalog_path}")
    print(f"Roster: {roster_path}")
    print("=" * 60)

    config = ShearExtractionConfig()
    profiles, diagnostics = extract_all_profiles(
        catalog_path, roster_path, config
    )

    # Summary
    n_ok = sum(1 for d in diagnostics.values() if d["status"] == "OK")
    n_fail = sum(1 for d in diagnostics.values() if d["status"] == "FAILED")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_ok} OK, {n_fail} FAILED out of {len(diagnostics)} filaments")

    if n_ok > 0:
        print(f"\nProfiles ready for mu_IR anchoring.")
        print(f"Next step: run rt_filament_forward_model.py with observed profiles")
    else:
        print(f"\nNo profiles extracted. Check catalog coverage and source density.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
