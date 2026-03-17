#!/usr/bin/env python3
"""
Full filament validation pipeline orchestrator.

Executes the complete freeze-then-test protocol:

1. Load frozen roster (v3, KiDS-overlap)
2. Extract observed WL shear profiles from KiDS-1000
3. Anchor mu_IR on training set (3 filaments)
4. Run ESG stability check (leave-one-out < 15%)
5. Freeze mu_IR — no further changes
6. Evaluate holdout set (8 filaments)
7. Adjudicate pass/fail per Contract Sec 8.1

Exit codes:
  0 = holdout PASS
  1 = holdout FAIL
  2 = infrastructure error (data missing, etc.)
  3 = ESG instability (mu_IR not anchored)
"""

import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from extract_wl_shear import (
    extract_all_profiles,
    ShearExtractionConfig,
)
from rt_filament_forward_model import (
    RTFilamentPipeline,
    RegimeCertificate,
    anchor_mu_IR,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).parent.parent
ROSTER_PATH = BASE_DIR / "rosters" / "filament_roster_v4.json"
CATALOG_PATH = BASE_DIR / "data" / "kids_lensing" / "KiDS_DR4.1_WL_gold_cat.fits"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Pass/fail gates (from Contract Sec 8.1)
PASS_FAIL_GATES = {
    "chi2_per_dof_max": 2.0,          # holdout chi2/dof < 2.0
    "sign_agreement_min_fraction": 0.75,  # >= 75% of holdout show correct sign
    "amplitude_ratio_range": (0.3, 3.0),  # predicted/observed within factor 3
    "cross_shear_max": 0.5,           # |gamma_x / gamma_t| < 0.5
    "esg_max_loo_change": 0.15,       # leave-one-out < 15%
    "first_jet_null_max": 0.01,       # first-Jet signal < 1% of second-Jet
    "scrambled_ratio_max": 0.15,      # scrambled/real < 15%
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def print_header(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


# ============================================================
# Main pipeline
# ============================================================

def main():
    timestamp = datetime.now(timezone.utc).isoformat()

    print_header("RT COSMIC FILAMENT VALIDATION PIPELINE")
    print(f"Timestamp: {timestamp}")
    print(f"Protocol: freeze-then-test (Contract v1.0)")

    # ----------------------------------------------------------
    # Step 0: Verify inputs exist
    # ----------------------------------------------------------
    print_header("STEP 0: Input verification")

    if not ROSTER_PATH.exists():
        print(f"FATAL: Roster not found: {ROSTER_PATH}")
        return 2

    if not CATALOG_PATH.exists():
        print(f"FATAL: KiDS catalog not found: {CATALOG_PATH}")
        print(f"Run download_kids_gold.py first.")
        return 2

    roster_hash = sha256_file(ROSTER_PATH)
    print(f"Roster: {ROSTER_PATH.name} (SHA-256: {roster_hash[:16]}...)")
    print(f"Catalog: {CATALOG_PATH.name} ({CATALOG_PATH.stat().st_size / 1e9:.2f} GB)")

    with open(ROSTER_PATH) as f:
        roster = json.load(f)
    n_train = len(roster.get("training", []))
    n_hold = len(roster.get("holdout", []))
    print(f"Roster: {n_train} training + {n_hold} holdout filaments")

    # ----------------------------------------------------------
    # Step 1: Extract observed WL shear profiles
    # ----------------------------------------------------------
    print_header("STEP 1: Extract observed tangential shear profiles")

    config = ShearExtractionConfig()
    profiles, diag = extract_all_profiles(
        CATALOG_PATH, ROSTER_PATH, config,
        output_path=OUTPUT_DIR / "observed_shear_profiles.json",
    )

    n_ok = sum(1 for d in diag.values() if d["status"] == "OK")
    if n_ok < n_train:
        print(f"FATAL: Only {n_ok} profiles extracted, need at least {n_train} (training)")
        return 2

    # Check we have training profiles
    training_ids = [str(f["filament_id"]) for f in roster["training"]]
    missing_train = [fid for fid in training_ids if fid not in profiles]
    if missing_train:
        print(f"FATAL: Missing training profiles: {missing_train}")
        return 2

    print(f"\nExtracted {n_ok} / {n_train + n_hold} profiles successfully")

    # ----------------------------------------------------------
    # Step 2: Anchor mu_IR on training set
    # ----------------------------------------------------------
    print_header("STEP 2: Anchor mu_IR on training filaments")

    # Prepare observed profiles dict for anchor_mu_IR
    observed_for_anchor = {}
    for fid in training_ids:
        if fid in profiles:
            r_bins, gamma_t, gamma_t_err = profiles[fid]
            observed_for_anchor[fid] = (r_bins, gamma_t, gamma_t_err)

    cert_initial = RegimeCertificate(mu_IR=1.0)  # starting guess

    anchor_result = anchor_mu_IR(
        str(ROSTER_PATH), observed_for_anchor, cert_initial
    )

    mu_IR_star = anchor_result["mu_IR_star"]
    esg_passed = anchor_result["ESG_passed"]
    max_loo = anchor_result["max_loo_change"]

    print(f"\n  mu_IR* = {mu_IR_star:.6f}")
    print(f"  chi2_min = {anchor_result['chi2_min']:.4f}")
    print(f"  ESG max LOO change = {max_loo:.4f} (threshold: {PASS_FAIL_GATES['esg_max_loo_change']})")
    print(f"  ESG passed: {esg_passed}")

    if not esg_passed:
        print(f"\nFATAL: ESG instability — mu_IR is not stable under leave-one-out.")
        print(f"  Max change: {max_loo:.4f} > {PASS_FAIL_GATES['esg_max_loo_change']}")

        # Save partial results
        with open(OUTPUT_DIR / "validation_result_FAILED_ESG.json", "w") as f:
            json.dump({
                "timestamp": timestamp,
                "status": "FAILED_ESG",
                "anchor_result": anchor_result,
            }, f, indent=2)
        return 3

    # ----------------------------------------------------------
    # Step 3: FREEZE mu_IR — no further changes
    # ----------------------------------------------------------
    print_header("STEP 3: FREEZE mu_IR")
    print(f"  mu_IR = {mu_IR_star:.6f}  [FROZEN]")
    print(f"  No further parameter changes permitted.")

    cert_frozen = RegimeCertificate(mu_IR=mu_IR_star)

    # ----------------------------------------------------------
    # Step 4: Run forward model on holdout with frozen parameters
    # ----------------------------------------------------------
    print_header("STEP 4: Holdout evaluation (8 filaments)")

    pipeline = RTFilamentPipeline(cert_frozen)
    holdout_results = []

    holdout_ids = [str(f["filament_id"]) for f in roster["holdout"]]

    for fil_data in roster["holdout"]:
        fid = str(fil_data["filament_id"])

        # Normalize roster entry
        fil_data_norm = dict(fil_data)
        fil_data_norm["roster_id"] = fid
        fil_data_norm["set"] = "holdout"

        filament = pipeline._build_filament_target(fil_data_norm)
        result = pipeline.run_single_filament(filament)
        result["null_tests"] = pipeline.run_null_tests(filament)

        # Compare with observed
        if fid in profiles:
            r_obs, gamma_obs, gamma_err = profiles[fid]
            result["observed_gamma_t"] = gamma_obs.tolist()
            result["observed_gamma_t_err"] = gamma_err.tolist()

            # Compute holdout chi2
            if result.get("Q_perp_profile") is not None:
                Q_pred = np.array(result["Q_perp_profile"])
                gamma_pred = cert_frozen.b_shear * Q_pred

                # Align binning (predicted uses symmetric, observed uses |r_perp|)
                # Use the positive half of predicted profile
                n_pred = len(Q_pred)
                n_obs = len(gamma_obs)

                # Simple: interpolate predicted onto observed bins
                r_pred = np.array(result["r_perp_bins"])
                r_pred_pos = r_pred[r_pred >= 0]
                Q_pred_pos = Q_pred[r_pred >= 0]

                from scipy.interpolate import interp1d
                if len(r_pred_pos) >= 2:
                    interp = interp1d(
                        r_pred_pos, Q_pred_pos,
                        bounds_error=False, fill_value=0.0
                    )
                    gamma_pred_at_obs = cert_frozen.b_shear * interp(r_obs)

                    mask_valid = gamma_err > 0
                    chi2 = np.sum(
                        ((gamma_pred_at_obs[mask_valid] - gamma_obs[mask_valid])
                         / gamma_err[mask_valid])**2
                    )
                    dof = np.sum(mask_valid) - 1  # -1 for mu_IR
                    result["holdout_chi2"] = float(chi2)
                    result["holdout_dof"] = int(dof)
                    result["holdout_chi2_per_dof"] = float(chi2 / dof) if dof > 0 else float('inf')

                    # Sign agreement
                    sign_match = np.sign(gamma_pred_at_obs[mask_valid]) == np.sign(gamma_obs[mask_valid])
                    result["sign_agreement"] = float(np.mean(sign_match))

                    # Amplitude ratio
                    A_pred = float(np.sum(np.abs(gamma_pred_at_obs[mask_valid])))
                    A_obs = float(np.sum(np.abs(gamma_obs[mask_valid])))
                    result["amplitude_ratio"] = A_pred / A_obs if A_obs > 0 else float('inf')

        holdout_results.append(result)

    # ----------------------------------------------------------
    # Step 5: Null test verification
    # ----------------------------------------------------------
    print_header("STEP 5: Null test verification")

    all_nulls_pass = True
    for result in holdout_results:
        fid = result["roster_id"]
        nulls = result.get("null_tests", {})

        fj = abs(nulls.get("first_jet_A_fil", 0))
        A = abs(result.get("A_fil", 1e-10))
        fj_ratio = fj / A if A > 0 else 0

        scram = abs(nulls.get("scrambled_A_fil", 0))
        scram_ratio = scram / A if A > 0 else 0

        print(f"  {fid}: 1J-null={fj_ratio:.4f}, scrambled={scram_ratio:.4f}")

        if fj_ratio > PASS_FAIL_GATES["first_jet_null_max"]:
            print(f"    FAIL: first-Jet null ratio {fj_ratio:.4f} > {PASS_FAIL_GATES['first_jet_null_max']}")
            all_nulls_pass = False
        if scram_ratio > PASS_FAIL_GATES["scrambled_ratio_max"]:
            print(f"    FAIL: scrambled ratio {scram_ratio:.4f} > {PASS_FAIL_GATES['scrambled_ratio_max']}")
            all_nulls_pass = False

    # ----------------------------------------------------------
    # Step 6: Adjudicate pass/fail
    # ----------------------------------------------------------
    print_header("STEP 6: VERDICT")

    # Collect holdout metrics
    chi2_per_dof_list = [r["holdout_chi2_per_dof"] for r in holdout_results
                         if "holdout_chi2_per_dof" in r]
    sign_list = [r["sign_agreement"] for r in holdout_results
                 if "sign_agreement" in r]
    amp_list = [r["amplitude_ratio"] for r in holdout_results
                if "amplitude_ratio" in r]

    gates = {}

    # Gate 1: chi2/dof
    if chi2_per_dof_list:
        mean_chi2 = np.mean(chi2_per_dof_list)
        gates["chi2_per_dof"] = {
            "value": float(mean_chi2),
            "threshold": PASS_FAIL_GATES["chi2_per_dof_max"],
            "passed": mean_chi2 < PASS_FAIL_GATES["chi2_per_dof_max"],
        }
    else:
        gates["chi2_per_dof"] = {"value": None, "passed": False, "reason": "no data"}

    # Gate 2: sign agreement
    if sign_list:
        mean_sign = np.mean(sign_list)
        gates["sign_agreement"] = {
            "value": float(mean_sign),
            "threshold": PASS_FAIL_GATES["sign_agreement_min_fraction"],
            "passed": mean_sign >= PASS_FAIL_GATES["sign_agreement_min_fraction"],
        }
    else:
        gates["sign_agreement"] = {"value": None, "passed": False, "reason": "no data"}

    # Gate 3: amplitude ratio
    if amp_list:
        lo, hi = PASS_FAIL_GATES["amplitude_ratio_range"]
        in_range = [lo <= a <= hi for a in amp_list]
        frac_ok = np.mean(in_range)
        gates["amplitude_ratio"] = {
            "fraction_in_range": float(frac_ok),
            "range": [lo, hi],
            "passed": frac_ok >= 0.75,
        }
    else:
        gates["amplitude_ratio"] = {"value": None, "passed": False, "reason": "no data"}

    # Gate 4: null tests
    gates["null_tests"] = {"passed": all_nulls_pass}

    # Gate 5: ESG stability (already checked)
    gates["esg_stability"] = {
        "max_loo_change": float(max_loo),
        "threshold": PASS_FAIL_GATES["esg_max_loo_change"],
        "passed": esg_passed,
    }

    # Overall verdict
    all_passed = all(g["passed"] for g in gates.values())
    verdict = "PASS" if all_passed else "FAIL"

    print(f"\n  Gate results:")
    for name, gate in gates.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"    {name}: {status} — {gate}")

    print(f"\n  ==============================")
    print(f"  VERDICT: {verdict}")
    print(f"  ==============================")

    # ----------------------------------------------------------
    # Save full results
    # ----------------------------------------------------------
    output = {
        "timestamp": timestamp,
        "verdict": verdict,
        "roster_hash": roster_hash,
        "mu_IR_frozen": mu_IR_star,
        "anchor_result": anchor_result,
        "gates": gates,
        "holdout_results": holdout_results,
        "pass_fail_criteria": PASS_FAIL_GATES,
        "regime_certificate": cert_frozen.__dict__,
    }

    # Serialize numpy types
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    output_clean = to_serializable(output)
    result_path = OUTPUT_DIR / f"validation_result_{verdict}.json"
    with open(result_path, "w") as f:
        json.dump(output_clean, f, indent=2)

    print(f"\nFull results saved: {result_path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
