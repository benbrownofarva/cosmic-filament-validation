#!/usr/bin/env python3
"""
Texture coupling constant g_T: zero-parameter prediction and matched-filter
extraction from cosmic filament weak-lensing data.

Two analyses:
  1. Zero-parameter prediction: apply g_T = 0.20226 (Particle Sector P49)
     and compare predicted gamma_t to KiDS-1000 observations.
  2. Matched-filter extraction: use predicted profile shape as template
     to extract g_T amplitude from data optimally.

Requires:
  - results/observed_shear_profiles.json (from extract_wl_shear.py)
  - Frozen roster (rosters/filament_roster_v4.json)

Run: python pipeline/g_T_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from rt_filament_forward_model import (
    RTFilamentPipeline,
    RegimeCertificate,
)

# ============================================================
# Configuration
# ============================================================

G_T_THEORY = 0.20226  # Particle Sector P49 branch-local estimate
B_SHEAR = 1.0         # deviatoric strain-optic coefficient

BASE_DIR = Path(__file__).parent.parent
ROSTER_PATH = BASE_DIR / "rosters" / "filament_roster_v4.json"
OBS_PATH = BASE_DIR / "results" / "observed_shear_profiles.json"
OUTPUT_PATH = BASE_DIR / "results" / "g_T_analysis.json"


# ============================================================
# Template generation
# ============================================================

def generate_templates(roster, cert):
    """Generate unit-coupling Q_perp templates for each filament."""
    pipeline = RTFilamentPipeline(cert)
    templates = {}

    all_filaments = roster.get("training", []) + roster.get("holdout", [])
    for fil_data in all_filaments:
        fid = str(fil_data.get("filament_id", "unknown"))
        fil_data_norm = dict(fil_data)
        fil_data_norm["roster_id"] = fid

        filament = pipeline._build_filament_target(fil_data_norm)
        result = pipeline.run_single_filament(filament)

        if result.get("Q_perp_profile") is not None:
            templates[fid] = {
                "r_bins": np.array(result["r_perp_bins"]),
                "Q_perp": np.array(result["Q_perp_profile"]),
                "A_fil": result["A_fil"],
            }

    return templates


# ============================================================
# Analysis 1: Zero-parameter prediction
# ============================================================

def zero_parameter_prediction(templates, obs_profiles, roster, g_T=G_T_THEORY):
    """Apply g_T from theory and compare to observed gamma_t."""
    results = {}
    training_ids = {str(f["filament_id"]) for f in roster.get("training", [])}

    for fid, tmpl in templates.items():
        if fid not in obs_profiles:
            continue

        r_obs = np.array(obs_profiles[fid]["r_perp_bins"])
        gamma_obs = np.array(obs_profiles[fid]["gamma_t"])
        gamma_err = np.array(obs_profiles[fid]["gamma_t_err"])

        # Interpolate template to observed bins (positive-r half)
        r_pos = tmpl["r_bins"][tmpl["r_bins"] >= 0]
        Q_pos = tmpl["Q_perp"][tmpl["r_bins"] >= 0]
        if len(r_pos) < 2:
            continue

        interp_func = interp1d(r_pos, Q_pos, bounds_error=False, fill_value=0.0)
        gamma_pred = g_T * B_SHEAR * interp_func(r_obs)

        mask = gamma_err > 0
        chi2 = float(np.sum(((gamma_pred[mask] - gamma_obs[mask]) / gamma_err[mask]) ** 2))
        dof = int(np.sum(mask))

        nonzero = (gamma_pred[mask] != 0) & (gamma_obs[mask] != 0)
        sign_agree = float(np.mean(
            np.sign(gamma_pred[mask][nonzero]) == np.sign(gamma_obs[mask][nonzero])
        )) if np.any(nonzero) else 0.0

        A_pred = float(np.sum(np.abs(gamma_pred[mask])))
        A_obs = float(np.sum(np.abs(gamma_obs[mask])))

        results[fid] = {
            "set": "training" if fid in training_ids else "holdout",
            "chi2": chi2,
            "dof": dof,
            "chi2_per_dof": chi2 / dof if dof > 0 else 0,
            "sign_agreement": sign_agree,
            "amplitude_ratio": A_pred / A_obs if A_obs > 0 else float("inf"),
            "mean_gamma_pred": float(np.mean(gamma_pred)),
            "mean_gamma_obs": float(np.mean(gamma_obs)),
        }

    return results


# ============================================================
# Analysis 2: Matched filter extraction
# ============================================================

def matched_filter_extraction(templates, obs_profiles, roster):
    """
    Extract g_T amplitude using predicted profile shape as template.

    Estimator:
        g_hat = sum(t * d / sigma^2) / sum(t^2 / sigma^2)

    where t = template (unit-coupling Q_perp), d = observed gamma_t,
    sigma = per-bin error. This is the minimum-variance unbiased estimator
    for the amplitude of a known signal shape in Gaussian noise.
    """
    per_filament = {}
    training_ids = {str(f["filament_id"]) for f in roster.get("training", [])}

    numerator_total = 0.0
    denominator_total = 0.0

    for fid, tmpl in templates.items():
        if fid not in obs_profiles:
            continue

        r_obs = np.array(obs_profiles[fid]["r_perp_bins"])
        gamma_obs = np.array(obs_profiles[fid]["gamma_t"])
        gamma_err = np.array(obs_profiles[fid]["gamma_t_err"])

        r_pos = tmpl["r_bins"][tmpl["r_bins"] >= 0]
        Q_pos = tmpl["Q_perp"][tmpl["r_bins"] >= 0]
        if len(r_pos) < 2:
            continue

        interp_func = interp1d(r_pos, Q_pos, bounds_error=False, fill_value=0.0)
        template = interp_func(r_obs)

        mask = (gamma_err > 0) & (template != 0)
        if np.sum(mask) < 3:
            continue

        t = template[mask]
        d = gamma_obs[mask]
        sigma = gamma_err[mask]

        num = np.sum(t * d / sigma ** 2)
        den = np.sum(t ** 2 / sigma ** 2)

        if den > 0:
            g_hat = num / den
            g_err = 1.0 / np.sqrt(den)

            per_filament[fid] = {
                "set": "training" if fid in training_ids else "holdout",
                "g_T_hat": float(g_hat),
                "g_T_err": float(g_err),
                "snr": float(g_hat / g_err),
            }

            numerator_total += num
            denominator_total += den

    # Global stacked estimate
    global_result = None
    if denominator_total > 0:
        g_global = numerator_total / denominator_total
        g_global_err = 1.0 / np.sqrt(denominator_total)
        global_result = {
            "g_T_hat": float(g_global),
            "g_T_err": float(g_global_err),
            "snr": float(g_global / g_global_err),
            "tension_with_theory_sigma": float(
                abs(g_global - G_T_THEORY) / g_global_err
            ),
            "n_filaments": len(per_filament),
        }

    return per_filament, global_result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("g_T Analysis: Zero-Parameter Prediction + Matched Filter")
    print(f"Theory value: g_T = {G_T_THEORY}")
    print("=" * 60)

    # Load inputs
    if not OBS_PATH.exists():
        print(f"Observed profiles not found: {OBS_PATH}")
        print("Run extract_wl_shear.py first.")
        return 1

    with open(OBS_PATH) as f:
        obs_data = json.load(f)
    obs_profiles = obs_data["profiles"]

    with open(ROSTER_PATH) as f:
        roster = json.load(f)

    # Generate templates
    print("\nGenerating unit-coupling templates...")
    cert = RegimeCertificate(mu_IR=1.0)
    templates = generate_templates(roster, cert)
    print(f"  {len(templates)} templates generated")

    # Analysis 1
    print("\n--- Analysis 1: Zero-parameter prediction (g_T = 0.202) ---")
    zp_results = zero_parameter_prediction(templates, obs_profiles, roster)

    chi2_vals = [r["chi2_per_dof"] for r in zp_results.values()]
    sign_vals = [r["sign_agreement"] for r in zp_results.values()]
    amp_vals = [r["amplitude_ratio"] for r in zp_results.values()]

    for fid, r in sorted(zp_results.items()):
        print(f"  {fid} ({r['set'][:5]}): chi2/dof={r['chi2_per_dof']:.2f}, "
              f"sign={r['sign_agreement']:.2f}, |pred/obs|={r['amplitude_ratio']:.3f}")
    print(f"\n  Mean: chi2/dof={np.mean(chi2_vals):.2f}, "
          f"sign={np.mean(sign_vals):.2f}, |pred/obs|={np.mean(amp_vals):.3f}")

    # Analysis 2
    print("\n--- Analysis 2: Matched filter extraction ---")
    mf_per_fil, mf_global = matched_filter_extraction(templates, obs_profiles, roster)

    for fid, r in sorted(mf_per_fil.items()):
        print(f"  {fid} ({r['set'][:5]}): g_T = {r['g_T_hat']:+.4f} +/- {r['g_T_err']:.4f} "
              f"(S/N = {r['snr']:+.2f})")

    if mf_global:
        print(f"\n  {'=' * 50}")
        print(f"  GLOBAL (stacked {mf_global['n_filaments']} filaments):")
        print(f"    g_T = {mf_global['g_T_hat']:.4f} +/- {mf_global['g_T_err']:.4f}")
        print(f"    S/N = {mf_global['snr']:.2f}")
        print(f"    Theory: {G_T_THEORY}")
        print(f"    Tension: {mf_global['tension_with_theory_sigma']:.2f} sigma")
        print(f"  {'=' * 50}")

    # Save
    output = {
        "analysis_date": "2026-03-22",
        "g_T_theory": G_T_THEORY,
        "g_T_source": "ParticleSector P49 branch-local estimate (gt_texture v0.9.0)",
        "zero_parameter_prediction": {
            "description": "Apply g_T = 0.20226 with no fitting.",
            "per_filament": zp_results,
            "summary": {
                "mean_chi2_per_dof": float(np.mean(chi2_vals)),
                "mean_sign_agreement": float(np.mean(sign_vals)),
                "mean_amplitude_ratio": float(np.mean(amp_vals)),
                "n_filaments": len(zp_results),
            },
        },
        "matched_filter_extraction": {
            "description": (
                "Extract g_T using predicted profile shape as matched-filter template. "
                "g_hat = sum(t*d/sigma^2) / sum(t^2/sigma^2)."
            ),
            "per_filament": mf_per_fil,
            "global_estimate": mf_global,
        },
        "amplitude_chain": {
            "chain": "geometry -> kappa*(mu_IR) -> eps_dev -> Q_perp(ell) -> gamma_t = g_T * b_shear * Q_perp",
            "frozen_parameters": cert.__dict__,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
