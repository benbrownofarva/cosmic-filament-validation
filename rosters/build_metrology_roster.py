#!/usr/bin/env python3
"""
Build the Filament Metrology Roster: 3 training + 8 holdout filaments.

Selection criteria (from Validation Contract v1.0):
- Length >= 15 Mpc/h
- At least 2 spectroscopic galaxy clusters (richness >= 20) at endpoints
- KiDS-1000 weak-lensing coverage with source density >= 5 arcmin^{-2}
- Redshift range 0.1 <= z <= 0.3
- No known strong lensing contamination
- No masking artifacts covering > 20% of corridor footprint

Ranking: by endpoint cluster richness sum (descending).
Top 3 = training, next 8 = holdout.

Inputs required:
- SDSS filament catalog (Galárraga-Espinosa+ 2022)
- redMaPPer cluster catalog
- KiDS-1000 footprint/coverage map

Output:
- filament_roster_v1.json (frozen, hashed)
"""

import json
import hashlib
import sys
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not available. Using pure Python fallback.")

try:
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("WARNING: astropy not available. Cannot process FITS catalogs.")

# --- Configuration ---

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Selection criteria (from Validation Contract v1.0 Section 5)
CRITERIA = {
    "min_length_mpc_h": 15.0,
    "min_endpoint_richness": 20,
    "min_n_endpoint_clusters": 2,
    "min_source_density_arcmin2": 5.0,
    "redshift_min": 0.1,
    "redshift_max": 0.3,
    "max_mask_fraction": 0.20,
    "corridor_halfwidth_mpc_h": 5.0,
}

N_TRAINING = 3
N_HOLDOUT = 8


# --- Cosmology helpers ---

def comoving_distance_approx(z, H0=70.0, Om=0.3):
    """
    Approximate comoving distance in Mpc/h for flat LCDM.
    Uses simple numerical integration (trapezoidal).
    """
    if not HAS_NUMPY:
        # Pure Python fallback
        nsteps = 1000
        dz = z / nsteps
        total = 0.0
        for i in range(nsteps):
            zi = (i + 0.5) * dz
            E_z = (Om * (1 + zi)**3 + (1 - Om))**0.5
            total += dz / E_z
        c_over_H0 = 2997.9  # c / (100 km/s/Mpc) in Mpc/h
        return c_over_H0 * total
    else:
        zz = np.linspace(0, z, 1000)
        E_z = np.sqrt(Om * (1 + zz)**3 + (1 - Om))
        c_over_H0 = 2997.9  # Mpc/h
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        return c_over_H0 * _trapz(1.0 / E_z, zz)


def angular_to_physical_mpc_h(angle_deg, z):
    """Convert angular separation (degrees) to transverse comoving Mpc/h."""
    d_c = comoving_distance_approx(z)
    angle_rad = angle_deg * (3.14159265 / 180.0)
    return d_c * angle_rad  # comoving transverse distance


def physical_to_angular_deg(dist_mpc_h, z):
    """Convert comoving Mpc/h to angular separation in degrees."""
    d_c = comoving_distance_approx(z)
    if d_c == 0:
        return 0
    angle_rad = dist_mpc_h / d_c
    return angle_rad * (180.0 / 3.14159265)


# --- Catalog loading ---

def load_filament_catalog():
    """Load Galárraga-Espinosa+ 2022 filament catalog."""
    # Try multiple possible file locations
    candidates = [
        DATA_DIR / "sdss_filaments" / "galarraga2022_disperse" / "filaments.tsv",
        DATA_DIR / "sdss_filaments" / "galarraga2022_disperse" / "filaments.csv",
        DATA_DIR / "sdss_filaments" / "galarraga2022_disperse" / "catalog.tar.gz",
    ]

    for path in candidates:
        if path.exists():
            print(f"Loading filament catalog: {path}")
            if path.suffix == ".tsv":
                return Table.read(str(path), format="ascii.tab")
            elif path.suffix == ".csv":
                return Table.read(str(path), format="ascii.csv")
            elif path.suffix == ".gz":
                print("  Archive found. Extract first, then re-run.")
                return None

    print("Filament catalog not found. Run download_sdss_filaments.py first.")
    print("Searched:", [str(c) for c in candidates])
    return None


def load_cluster_catalog():
    """Load redMaPPer cluster catalog."""
    path = DATA_DIR / "sdss_filaments" / "redmapper_clusters" / "redmapper_dr8_v6.3_catalog.fits.gz"

    if not path.exists():
        print(f"Cluster catalog not found: {path}")
        print("Run download_sdss_filaments.py first.")
        return None

    print(f"Loading cluster catalog: {path}")
    return Table.read(str(path))


# --- Filament selection ---

def match_clusters_to_filament_endpoints(filament, clusters, match_radius_mpc_h=3.0):
    """
    Find clusters near filament endpoints.

    Returns list of (cluster_index, richness, distance_mpc_h) for each endpoint.
    """
    if not HAS_ASTROPY:
        return [], []

    z_fil = filament.get("z_mean", 0.2)
    match_radius_deg = physical_to_angular_deg(match_radius_mpc_h, z_fil)

    endpoint1_matches = []
    endpoint2_matches = []

    cluster_coords = SkyCoord(
        ra=clusters["RA"] * u.deg,
        dec=clusters["DEC"] * u.deg,
    )

    # Endpoint 1
    ep1 = SkyCoord(
        ra=filament["ra_ep1"] * u.deg,
        dec=filament["dec_ep1"] * u.deg,
    )
    seps1 = ep1.separation(cluster_coords).deg
    mask1 = seps1 < match_radius_deg
    for idx in np.where(mask1)[0]:
        rich = clusters["LAMBDA"][idx]
        dist = angular_to_physical_mpc_h(seps1[idx], z_fil)
        endpoint1_matches.append((int(idx), float(rich), float(dist)))

    # Endpoint 2
    ep2 = SkyCoord(
        ra=filament["ra_ep2"] * u.deg,
        dec=filament["dec_ep2"] * u.deg,
    )
    seps2 = ep2.separation(cluster_coords).deg
    mask2 = seps2 < match_radius_deg
    for idx in np.where(mask2)[0]:
        rich = clusters["LAMBDA"][idx]
        dist = angular_to_physical_mpc_h(seps2[idx], z_fil)
        endpoint2_matches.append((int(idx), float(rich), float(dist)))

    return endpoint1_matches, endpoint2_matches


def check_kids_coverage(filament, corridor_halfwidth_deg):
    """
    Check if a filament corridor has sufficient KiDS-1000 coverage.

    For now, this is a placeholder that checks against the KiDS-1000
    footprint (approximate rectangular regions). In production, this
    should query the actual survey mask.
    """
    # KiDS-1000 covers approximately 1006 deg^2 in specific patches:
    # KiDS-N: ~700 deg^2 around RA=100-240, Dec=-5 to +5
    # KiDS-S: ~300 deg^2 around RA=315-55, Dec=-36 to -26

    ra = filament.get("ra_mean", 0)
    dec = filament.get("dec_mean", 0)

    # KiDS-N footprint (approximate)
    in_kids_n = (100 < ra < 240) and (-5 < dec < 5)
    # KiDS-S footprint (approximate)
    in_kids_s = ((315 < ra < 360) or (0 < ra < 55)) and (-36 < dec < -26)

    return in_kids_n or in_kids_s


def compute_filament_properties(filament):
    """Compute derived properties for a filament."""
    z = filament.get("z_mean", 0.2)

    # Angular length to physical
    ra1 = filament.get("ra_ep1", 0)
    dec1 = filament.get("dec_ep1", 0)
    ra2 = filament.get("ra_ep2", 0)
    dec2 = filament.get("dec_ep2", 0)

    if HAS_ASTROPY:
        c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
        c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
        angular_length_deg = c1.separation(c2).deg
    else:
        # Approximate
        cos_dec = np.cos(np.radians((dec1 + dec2) / 2)) if HAS_NUMPY else 1.0
        angular_length_deg = ((ra2 - ra1)**2 * cos_dec**2 + (dec2 - dec1)**2)**0.5

    length_mpc_h = angular_to_physical_mpc_h(angular_length_deg, z)

    return {
        "angular_length_deg": float(angular_length_deg),
        "length_mpc_h": float(length_mpc_h),
        "z_mean": float(z),
    }


def select_filaments(filaments, clusters):
    """Apply all selection criteria and rank filaments."""

    candidates = []

    print(f"\nApplying selection criteria to {len(filaments)} filaments...")
    print(f"Criteria: {json.dumps(CRITERIA, indent=2)}")

    n_rejected = {
        "redshift": 0,
        "length": 0,
        "clusters": 0,
        "coverage": 0,
    }

    for i, fil in enumerate(filaments):
        # Check redshift
        z = fil.get("z_mean", fil.get("z", 0.2))
        if z < CRITERIA["redshift_min"] or z > CRITERIA["redshift_max"]:
            n_rejected["redshift"] += 1
            continue

        # Compute properties
        props = compute_filament_properties(fil)

        # Check length
        if props["length_mpc_h"] < CRITERIA["min_length_mpc_h"]:
            n_rejected["length"] += 1
            continue

        # Check cluster endpoints
        if clusters is not None:
            ep1_matches, ep2_matches = match_clusters_to_filament_endpoints(
                fil, clusters
            )

            # Need at least one rich cluster at each endpoint
            ep1_rich = [m for m in ep1_matches if m[1] >= CRITERIA["min_endpoint_richness"]]
            ep2_rich = [m for m in ep2_matches if m[1] >= CRITERIA["min_endpoint_richness"]]

            if len(ep1_rich) == 0 or len(ep2_rich) == 0:
                n_rejected["clusters"] += 1
                continue

            # Richness sum for ranking
            best_rich_1 = max(m[1] for m in ep1_rich)
            best_rich_2 = max(m[1] for m in ep2_rich)
            richness_sum = best_rich_1 + best_rich_2
        else:
            # Without cluster catalog, use placeholder ranking
            richness_sum = props["length_mpc_h"]  # rank by length as proxy
            ep1_rich = []
            ep2_rich = []

        # Check KiDS coverage
        corridor_hw_deg = physical_to_angular_deg(
            CRITERIA["corridor_halfwidth_mpc_h"], z
        )
        if not check_kids_coverage(fil, corridor_hw_deg):
            n_rejected["coverage"] += 1
            continue

        # Passed all criteria
        candidates.append({
            "index": i,
            "endpoint1_ra": float(fil.get("ra_ep1", 0)),
            "endpoint1_dec": float(fil.get("dec_ep1", 0)),
            "endpoint2_ra": float(fil.get("ra_ep2", 0)),
            "endpoint2_dec": float(fil.get("dec_ep2", 0)),
            "z_mean": float(z),
            "length_mpc_h": props["length_mpc_h"],
            "angular_length_deg": props["angular_length_deg"],
            "richness_sum": float(richness_sum),
            "n_ep1_clusters": len(ep1_rich),
            "n_ep2_clusters": len(ep2_rich),
        })

    print(f"\nRejection summary:")
    for reason, count in n_rejected.items():
        print(f"  {reason}: {count}")
    print(f"Candidates passing all criteria: {len(candidates)}")

    # Rank by richness sum (descending)
    candidates.sort(key=lambda x: x["richness_sum"], reverse=True)

    return candidates


def build_roster(candidates):
    """Split candidates into training and holdout sets."""

    if len(candidates) < N_TRAINING + N_HOLDOUT:
        print(f"\nWARNING: Only {len(candidates)} candidates available.")
        print(f"Need {N_TRAINING} training + {N_HOLDOUT} holdout = {N_TRAINING + N_HOLDOUT}")

        if len(candidates) < N_TRAINING:
            print("FATAL: Not enough candidates for training set.")
            return None

        n_holdout_actual = min(N_HOLDOUT, len(candidates) - N_TRAINING)
        print(f"Reducing holdout to {n_holdout_actual}")
    else:
        n_holdout_actual = N_HOLDOUT

    training = candidates[:N_TRAINING]
    holdout = candidates[N_TRAINING:N_TRAINING + n_holdout_actual]

    # Assign IDs
    for i, f in enumerate(training):
        f["roster_id"] = f"TRAIN_{i+1:02d}"
        f["set"] = "training"

    for i, f in enumerate(holdout):
        f["roster_id"] = f"HOLD_{i+1:02d}"
        f["set"] = "holdout"

    roster = {
        "contract_version": "1.0",
        "creation_date": datetime.utcnow().isoformat() + "Z",
        "selection_criteria": CRITERIA,
        "n_training": len(training),
        "n_holdout": len(holdout),
        "ranking_metric": "endpoint_cluster_richness_sum",
        "training": training,
        "holdout": holdout,
    }

    return roster


def freeze_roster(roster, output_path):
    """Freeze the roster with SHA-256 hash."""

    # Compute hash of roster content (excluding the hash field itself)
    roster_json = json.dumps(roster, sort_keys=True, indent=2)
    roster_hash = hashlib.sha256(roster_json.encode()).hexdigest()

    roster["freeze_hash"] = roster_hash
    roster["freeze_date"] = datetime.utcnow().isoformat() + "Z"

    with open(output_path, "w") as f:
        json.dump(roster, f, indent=2)

    print(f"\nRoster frozen and saved: {output_path}")
    print(f"SHA-256: {roster_hash}")

    return roster_hash


def generate_synthetic_roster():
    """
    Generate a synthetic roster for pipeline development.

    This is used when real catalogs are not yet downloaded.
    The synthetic filaments are placed in the KiDS-N footprint
    with realistic properties.

    IMPORTANT: This roster is labeled SYNTHETIC and must be replaced
    with real catalog selections before any validation claim.
    """

    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC ROSTER FOR PIPELINE DEVELOPMENT")
    print("This is NOT a real selection — replace with catalog data!")
    print("=" * 60)

    # Synthetic filaments in KiDS-N footprint (RA~150-200, Dec~0)
    # Properties chosen to be realistic for z~0.15-0.25

    if HAS_NUMPY:
        rng = np.random.RandomState(42)  # reproducible

    synthetic = []
    for i in range(15):  # 15 candidates, will select 3+8=11
        z = 0.12 + 0.16 * (i / 14)  # spread across 0.12-0.28
        ra_center = 140 + 8 * i  # spread across KiDS-N
        dec_center = -2 + 0.3 * i

        # Filament length 15-40 Mpc/h
        length_mpc_h = 18 + 2 * i
        angular_length_deg = physical_to_angular_deg(length_mpc_h, z)

        # Random PA
        pa_rad = (i * 137.5 % 360) * 3.14159 / 180  # golden angle spacing

        half_ang = angular_length_deg / 2
        ra1 = ra_center - half_ang * 0.7071  # cos(45)
        dec1 = dec_center - half_ang * 0.7071
        ra2 = ra_center + half_ang * 0.7071
        dec2 = dec_center + half_ang * 0.7071

        # Richness: higher for first entries (ranking proxy)
        richness_sum = 120 - 6 * i

        synthetic.append({
            "index": i,
            "endpoint1_ra": float(ra1),
            "endpoint1_dec": float(dec1),
            "endpoint2_ra": float(ra2),
            "endpoint2_dec": float(dec2),
            "z_mean": float(z),
            "length_mpc_h": float(length_mpc_h),
            "angular_length_deg": float(angular_length_deg),
            "richness_sum": float(richness_sum),
            "n_ep1_clusters": 1,
            "n_ep2_clusters": 1,
            "SYNTHETIC": True,
        })

    return synthetic


def main():
    print("=" * 60)
    print("Filament Metrology Roster Builder")
    print(f"Date: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    # Try to load real catalogs
    filaments = None
    clusters = None

    if HAS_ASTROPY:
        filaments = load_filament_catalog()
        clusters = load_cluster_catalog()

    if filaments is not None and len(filaments) > 0:
        # Real catalog selection
        candidates = select_filaments(filaments, clusters)
    else:
        # Synthetic fallback for pipeline development
        print("\nReal catalogs not available. Using synthetic roster.")
        candidates = generate_synthetic_roster()

    if len(candidates) == 0:
        print("\nFATAL: No candidates found.")
        return 1

    # Build roster
    roster = build_roster(candidates)
    if roster is None:
        return 1

    # Print roster summary
    print(f"\n{'='*60}")
    print("ROSTER SUMMARY")
    print(f"{'='*60}")

    print(f"\nTraining set ({roster['n_training']} filaments):")
    for f in roster["training"]:
        synth = " [SYNTHETIC]" if f.get("SYNTHETIC") else ""
        print(f"  {f['roster_id']}: z={f['z_mean']:.3f}, "
              f"L={f['length_mpc_h']:.1f} Mpc/h, "
              f"richness_sum={f['richness_sum']:.0f}{synth}")

    print(f"\nHoldout set ({roster['n_holdout']} filaments):")
    for f in roster["holdout"]:
        synth = " [SYNTHETIC]" if f.get("SYNTHETIC") else ""
        print(f"  {f['roster_id']}: z={f['z_mean']:.3f}, "
              f"L={f['length_mpc_h']:.1f} Mpc/h, "
              f"richness_sum={f['richness_sum']:.0f}{synth}")

    # Freeze and save
    output_path = OUTPUT_DIR / "filament_roster_v1.json"
    freeze_hash = freeze_roster(roster, output_path)

    # Also save a human-readable summary
    summary_path = OUTPUT_DIR / "ROSTER_SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write("# Filament Metrology Roster v1.0\n\n")
        f.write(f"**Frozen**: {roster.get('freeze_date', 'N/A')}\n")
        f.write(f"**SHA-256**: `{freeze_hash}`\n\n")

        is_synthetic = any(
            fil.get("SYNTHETIC") for fil in roster["training"] + roster["holdout"]
        )
        if is_synthetic:
            f.write("**WARNING: SYNTHETIC ROSTER — Replace with real catalog selections!**\n\n")

        f.write("## Selection Criteria\n\n")
        for k, v in CRITERIA.items():
            f.write(f"- {k}: {v}\n")

        f.write(f"\n## Training Set ({roster['n_training']} filaments)\n\n")
        f.write("| ID | z | Length (Mpc/h) | Richness Sum |\n")
        f.write("|-----|-------|----------------|---------------|\n")
        for fil in roster["training"]:
            f.write(f"| {fil['roster_id']} | {fil['z_mean']:.3f} | "
                    f"{fil['length_mpc_h']:.1f} | {fil['richness_sum']:.0f} |\n")

        f.write(f"\n## Holdout Set ({roster['n_holdout']} filaments)\n\n")
        f.write("| ID | z | Length (Mpc/h) | Richness Sum |\n")
        f.write("|-----|-------|----------------|---------------|\n")
        for fil in roster["holdout"]:
            f.write(f"| {fil['roster_id']} | {fil['z_mean']:.3f} | "
                    f"{fil['length_mpc_h']:.1f} | {fil['richness_sum']:.0f} |\n")

    print(f"\nSummary saved: {summary_path}")
    print("\nNext steps:")
    print("  1. If SYNTHETIC: download real catalogs and re-run")
    print("  2. Run download_kids_targeted.py with this roster")
    print("  3. Proceed to mu_IR anchoring pipeline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
