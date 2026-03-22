# Cosmic Filament Weak-Lensing Validation Dataset

**A curated dataset of SDSS cosmic filaments with KiDS-1000 weak-lensing shear profiles for filament-scale gravity tests.**

| Item | Value |
|------|-------|
| Source catalog | Tempel, Stoica & Saar 2014, MNRAS 438, 3465 (SDSS DR8 Bisous) |
| WL survey | KiDS-1000 DR4.1 (Giblin+ 2021, Kuijken+ 2019) |
| Filament candidates | 558 (all tiers) |
| Primary training set (Tier A) | 19 filaments in KiDS-North footprint |
| Relaxed training set (Tier A+B) | 28 filaments in KiDS-North footprint |
| Observed shear profiles | 11 filaments with extracted gamma_t(r_perp) |
| g_T matched-filter result | 0.54 +/- 0.33 (1.7-sigma, consistent with theory at 1.05-sigma) |
| Dataset freeze date | 2026-03-17 |

**See [ANALYSIS.md](ANALYSIS.md) for the full g_T measurement writeup.**

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy scipy astropy requests

# 2. Download the SDSS filament catalog (53 MB)
python data/download_sdss_filaments.py

# 3. Download KiDS-1000 gold WL catalog (16.5 GB)
python data/download_kids_gold.py
# or: wget -c 'https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits'

# 4. Run the forward model (no WL data needed)
cd pipeline
python rt_filament_forward_model.py

# 5. Extract observed WL profiles (requires KiDS catalog)
python extract_wl_shear.py

# 6. Run full validation
python run_full_validation.py
```

---

## 1. Scientific Context

Cosmic filaments — the elongated bridges connecting galaxy clusters in the cosmic
web — produce a weak gravitational lensing signal that can be measured by stacking
background galaxy shapes along the filament spine. This dataset enables testing
whether a given gravity model correctly predicts the **transverse tangential shear
profile** gamma_t(r_perp) around filaments, using only observed baryonic geometry
as input.

The primary observable is the **Stacked Filament Transverse Shear Profile (SFTSP)**:
the mean tangential shear measured perpendicular to the filament spine, binned in
transverse distance r_perp from 0 to 5 Mpc/h.

**Reference for filament lensing methodology**: de Graaff+ 2019, A&A 624, A48
(detected filament lensing at 3.4-sigma by stacking ~5000 SDSS filaments with
KiDS-450 WL data).

---

## 2. Filament Selection Criteria

### 2.1 Source catalog

The filaments come from the Tempel, Stoica & Saar (2014) Bisous catalog applied to
SDSS DR8. The catalog contains **15,421 filaments** identified by the Bisous
stochastic marked-point-process method, with associated galaxy members (576,493
galaxies total).

- **Download**: `python data/download_sdss_filaments.py`
- **Original source**: https://www.aai.ee/~elmo/sdss-filaments/dr8_filaments.fits
- **Reference**: Tempel, Stoica, Saar 2014, MNRAS 438, 3465

The FITS file contains three HDUs:
| HDU | Name | Rows | Description |
|-----|------|------|-------------|
| 1 | FILAMENTS | 15,421 | Filament properties (id, length, endpoint richness) |
| 2 | FILPOINTS | 275,599 | Spine points (x, y, z in SGX/SGY/SGZ Mpc/h) |
| 3 | GALAXIES | 576,493 | Member galaxies (RA, Dec, redshift, magnitudes) |

### 2.2 Selection funnel

Starting from 15,421 filaments, the following cuts are applied sequentially:

| Cut | Criterion | Remaining | Eliminated |
|-----|-----------|-----------|------------|
| 0 | All filaments with galaxy members | 15,421 | 0 |
| 1 | Length >= 15 Mpc/h | 2,306 | 13,115 |
| 2 | Mean redshift 0.1 <= z <= 0.2 | 600 | 1,706 |
| 3 | Angular separation >= 0.5 deg | 599 | 1 |
| 4a | Endpoint richness >= 15 (both) | 354 | 245 |
| 4b | Endpoint richness >= 10 (both) | 558 | 41 |
| 5 | KiDS-North footprint overlap | **19** (Tier A) / **28** (A+B) | 335 / 530 |

### 2.3 Rationale for each cut

**Cut 1 — Length >= 15 Mpc/h**: Filaments shorter than 15 Mpc/h subtend less than
~0.5 degrees at z=0.15, making transverse profile extraction unreliable with 0.25
Mpc/h bins. The 15 Mpc/h threshold ensures at least 3 resolution elements across
the corridor width (R_max = 5 Mpc/h) at the readout scale (ell = 2 Mpc/h).

**Cut 2 — Redshift 0.1 <= z <= 0.2**: The lower bound z=0.1 ensures sufficient
angular diameter distance (~300 Mpc/h) for the transverse profile to subtend a
measurable angle (~1 deg for R_max = 5 Mpc/h). The upper bound z=0.2 ensures
the SDSS spectroscopic sample is approximately volume-complete and the Bisous
filament detection is reliable.

**Cut 3 — Angular separation >= 0.5 deg**: Filament endpoints that are angularly
close (< 0.5 deg) may be artefacts of the Bisous algorithm or represent structures
too compact for transverse profile extraction. This cut removes 1 filament.

**Cut 4a — Endpoint richness >= 15 (Tier A primary)**: Each filament endpoint must
be associated with at least 15 galaxies, ensuring robust endpoint localization. The
endpoint position is computed as the mean RA/Dec of the 20% of member galaxies
closest to (4a) or farthest from (4b) the filament start in spine-point ordering.

**Cut 4b — Endpoint richness >= 10 (Tier B relaxed)**: Relaxed version allowing
endpoints with 10-14 galaxies. These filaments have less precise endpoint
localization but may contribute useful signal in stacking.

**Cut 5 — KiDS-North footprint**: Both endpoints must fall within the verified
KiDS-1000 North equatorial strip: **RA [130, 235] deg, Dec [-3, +2.5] deg**. This
footprint was determined empirically from the KiDS-1000 DR4.1 gold WL catalog
(21,262,011 sources; source density drops sharply outside this region).

### 2.4 Endpoint extraction method

Filament endpoints are not directly stored in the Bisous catalog. We extract them
from the member galaxies:

1. Select all galaxies with `fil_id == filament_id`
2. Sort galaxies by `fil_idpts` (the spine-point index assigned by Bisous)
3. Define endpoint 1 as the mean position of the first 20% of sorted galaxies
   (minimum 5 galaxies)
4. Define endpoint 2 as the mean position of the last 20% of sorted galaxies
   (minimum 5 galaxies)
5. Convert to equatorial coordinates (RA, Dec) using the galaxy catalog positions

**Important**: The `fil_idpts` values are global indices into the FILPOINTS table,
not zero-based local indices. Sorting by `fil_idpts` correctly orders galaxies
along the filament spine.

### 2.5 Tier classification

Each filament is assigned to exactly one tier:

| Tier | Criteria | Count | Use case |
|------|----------|-------|----------|
| **A_primary** | All cuts including ngal>=15 + KiDS | 19 | Primary training/validation |
| **B_relaxed** | All cuts with ngal>=10 (not 15) + KiDS | 9 | Extended training set |
| **C_allsky** | ngal>=15 but outside KiDS footprint | 335 | Future surveys (Euclid, Rubin) |
| **D_allsky_relaxed** | ngal>=10 but outside KiDS footprint | 195 | Future surveys, lower quality |

---

## 3. Weak-Lensing Data

### 3.1 KiDS-1000 gold catalog

The weak-lensing shape measurements come from the KiDS-1000 DR4.1 gold sample:

| Property | Value |
|----------|-------|
| Catalog | `KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits` |
| Size | 16.5 GB (17,712,469,440 bytes) |
| Sources | 21,262,011 |
| After quality cuts | 21,092,189 (99.2%) |
| Columns used | ALPHA_J2000, DELTA_J2000, e1, e2, weight, Z_B, fitclass |
| Reference | Giblin+ 2021, A&A 645, A105 |
| Download URL | https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits |

**Quality cuts applied**:
- `fitclass == 0` (reliable star/galaxy separation)
- `weight > 0` (nonzero lensfit weight)
- `Z_B < 1.2` (reliable photometric redshift)
- Finite e1, e2 values

### 3.2 Shear extraction method

For each filament, the tangential shear profile gamma_t(r_perp) is extracted as follows:

1. **Background selection**: Sources with `Z_B > z_filament + 0.1` (ensures sources
   are behind the filament lens plane)

2. **Corridor selection**: Sources within a rectangular region along the filament
   spine, width = 2 * R_max = 10 Mpc/h, length = filament length + 4 Mpc/h buffer
   at each end

3. **Tangential shear**: For each source at position angle phi relative to the
   filament perpendicular direction:
   ```
   gamma_t = -(e1 * cos(2*phi) + e2 * sin(2*phi))
   gamma_x = +(e1 * sin(2*phi) - e2 * cos(2*phi))
   ```

4. **Binning**: 20 bins in |r_perp| from 0 to R_max = 5 Mpc/h (bin width = 0.25 Mpc/h)

5. **Weighting**: Each source receives combined weight = lensfit_weight * Sigma_crit^{-1}(z_lens, z_source),
   where Sigma_crit is the critical surface density for gravitational lensing

6. **Errors**: Shape noise error per bin = sigma_e / sqrt(N_eff), with sigma_e = 0.28
   per component (KiDS-1000 shape noise) and N_eff = (sum w)^2 / (sum w^2)

7. **Cross-component**: gamma_x (B-mode) is computed simultaneously as a null
   diagnostic; it should be consistent with zero for a pure lensing signal

### 3.3 Angular-to-physical conversion

All angular separations are converted to physical (comoving) distances using:
- Flat LCDM cosmology with Omega_m = 0.3, h = 1.0 (distances in Mpc/h)
- Angular diameter distance: d_A(z) = d_C(z) / (1 + z)
- Comoving distance: d_C(z) = (c/H_0) * integral_0^z dz' / E(z')
- E(z) = sqrt(Omega_m * (1+z)^3 + (1 - Omega_m))

---

## 4. Observed Shear Profiles

Tangential shear profiles have been extracted for all 11 filaments in the v4 roster
(3 training + 8 holdout from Tier A). Results are in `results/observed_shear_profiles.json`.

### 4.1 Per-filament summary

| ID | Set | z | N_sources | mean(gamma_t) | cumulative S/N |
|----|-----|---|-----------|----------------|----------------|
| 84 | Training | 0.107 | 224,531 | +0.001453 | 6.4 |
| 282 | Training | 0.110 | 244,852 | -0.001067 | 4.1 |
| 4506 | Training | 0.113 | 75,169 | +0.000157 | 3.4 |
| 4764 | Holdout | 0.127 | 147,641 | +0.000471 | 3.5 |
| 86 | Holdout | 0.110 | 271,462 | -0.000259 | 4.3 |
| 7830 | Holdout | 0.116 | 154,707 | -0.001447 | 4.4 |
| 2104 | Holdout | 0.121 | 141,794 | -0.000313 | 4.9 |
| 2316 | Holdout | 0.121 | 147,794 | -0.002015 | 4.7 |
| 3585 | Holdout | 0.125 | 85,112 | +0.002364 | 4.2 |
| 3158 | Holdout | 0.139 | 181,200 | -0.001305 | 4.9 |
| 1971 | Holdout | 0.105 | 141,432 | +0.000069 | 4.4 |

### 4.2 Stacked signal

Inverse-variance-weighted stack of all 11 filaments:
- **Stacked mean gamma_t**: -0.000286 +/- 0.001034
- **Stacked S/N**: 3.9
- **Per-bin S/N range**: -2.1 to +1.7

The stacked signal is marginal (3.9-sigma across 20 bins), consistent with the
filament lensing literature where ~5000 filaments are needed for a robust detection
(de Graaff+ 2019).

---

## 5. Forward Model Pipeline

The `pipeline/` directory contains a forward model that predicts the filament
transverse shear profile from first principles:

### 5.1 Pipeline steps

```
baryonic geometry --> closure solve (kappa*) --> log-strain (eps*)
    --> Liouville moments --> Q(x; ell) --> SFTSP --> A_fil
```

| Step | Module | Description |
|------|--------|-------------|
| A | `NonlocalClosureSolver.solve_kappa()` | Solve screened Poisson PDE for co-metric kappa* |
| B | `compute_log_strain()` | Extract bulk (theta) and deviatoric (eps^dev) strain |
| D | `compute_Q_anisotropy()` | Smooth deviatoric strain at readout scale ell |
| D | `extract_transverse_profile()` | Project Q onto filament perpendicular direction |
| D | `compute_integrated_amplitude()` | Integrate A_fil = int Q_perp * r_perp dr_perp |

### 5.2 Regime certificate (frozen parameters)

| Parameter | Symbol | Value | Status |
|-----------|--------|-------|--------|
| Readout scale | ell | 2.0 Mpc/h | Frozen |
| Corridor half-width | R_max | 5.0 Mpc/h | Frozen |
| Kernel sigma | sigma | 1.414 Mpc/h | Frozen |
| Bulk modulus | lambda_IR | 1.0 | Derived |
| Shear modulus | mu_IR | **pending** | To be anchored |
| Gradient penalty | alpha | 0.01 | Derived |
| Bulk strain-optic | a_bulk | 1.0 | Fixed |
| Deviatoric strain-optic | b_shear | 1.0 | Fixed |
| Grid resolution | dx | 0.5 Mpc/h | Numerical |

### 5.3 Null test battery

Four null tests are implemented, all passing with placeholder mu_IR = 1.0:

| Null test | Method | Gate | Result |
|-----------|--------|------|--------|
| First-Jet (mu_IR=0) | Kill shear channel | A_fil^null < 1% of full | **PASS** (0.0 for all) |
| Isotropy | Isotropic average of Q | A_fil^null < 10% of full | **PASS** (0.0 by construction) |
| Scrambled geometry | Randomize source positions | A_fil^null < 15% of full | **PASS** (~1.5% for all) |
| Scale robustness | Recompute at 0.7x and 1.4x ell | Same sign, within [0.3, 3.0] | **PASS** |

### 5.4 mu_IR anchoring protocol

The shear modulus mu_IR is the **single free parameter** to be anchored on the
training set before holdout evaluation:

1. Minimize chi^2 between predicted Q_perp and observed gamma_t over 3 training
   filaments, searching log(mu_IR) in [log(0.01), log(10.0)]
2. ESG stability check: leave-one-out must change mu_IR by less than 15%
3. If ESG passes, freeze mu_IR and evaluate 8 holdout filaments with no refit

**Current status**: ESG FAILED (90% leave-one-out variation). The per-filament S/N
is insufficient for stable 3-filament anchoring at the observed signal level
(gamma_t ~ 10^-3).

---

## 6. Repository Structure

```
.
├── README.md                          # This file
├── ANALYSIS.md                        # g_T measurement writeup
├── contracts/
│   └── COSMIC_FILAMENT_VALIDATION_CONTRACT_v1.md  # Pre-registered protocol
├── data/
│   ├── filament_candidates.json       # 558 candidates with tier + criteria flags
│   ├── filament_candidates.csv        # Same data as CSV
│   ├── download_sdss_filaments.py     # Download SDSS DR8 Bisous catalog (53 MB)
│   └── download_kids_gold.py          # Download KiDS-1000 gold catalog (16.5 GB)
├── rosters/
│   ├── filament_roster_v4.json        # Frozen roster: 3 training + 8 holdout (Tier A)
│   └── build_metrology_roster.py      # Roster construction script
├── pipeline/
│   ├── rt_filament_forward_model.py   # Forward model (Steps A-D)
│   ├── extract_wl_shear.py           # KiDS-1000 WL shear extraction
│   ├── run_full_validation.py         # End-to-end validation orchestrator
│   └── g_T_analysis.py               # g_T zero-param prediction + matched filter
└── results/
    ├── observed_shear_profiles.json   # Extracted gamma_t profiles (11 filaments)
    ├── pipeline_results_dev.json      # Forward model predictions (mu_IR=1.0)
    ├── selection_funnel.json          # Selection funnel summary
    ├── g_T_analysis.json              # g_T measurement results
    └── validation_analysis_INCONCLUSIVE.json  # Validation verdict and diagnostics
```

---

## 7. Reproducing the Analysis

### 7.1 Environment

```
Python 3.12
numpy >= 1.24
scipy >= 1.10
astropy >= 5.3
requests (for downloads only)
```

### 7.2 Step-by-step reproduction

**Step 1: Download source data**
```bash
python data/download_sdss_filaments.py
# Downloads dr8_filaments.fits (52.8 MB) from aai.ee
```

**Step 2: Verify filament catalog**
```python
from astropy.io import fits
hdul = fits.open('data/sdss_filaments/dr8_filaments.fits')
assert len(hdul['FILAMENTS'].data) == 15421
assert len(hdul['GALAXIES'].data) == 576493
```

**Step 3: Verify selection funnel**
The file `data/filament_candidates.json` contains 558 filaments with per-filament
boolean flags for each selection criterion. To verify:
```python
import json
with open('data/filament_candidates.json') as f:
    d = json.load(f)
tier_A = [f for f in d['filaments'] if f['tier'] == 'A_primary']
assert len(tier_A) == 19  # Primary training set
```

**Step 4: Download KiDS-1000 catalog** (16.5 GB)
```bash
python data/download_kids_gold.py
# or wget directly (supports resume):
wget -c 'https://kids.strw.leidenuniv.nl/DR4/data_files/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits' \
    -O data/kids_lensing/KiDS_DR4.1_WL_gold_cat.fits
```

**Step 5: Extract WL shear profiles**
```bash
cd pipeline
python extract_wl_shear.py
# Reads KiDS catalog, extracts gamma_t(r_perp) for all 11 roster filaments
# Output: results/observed_shear_profiles.json
```

**Step 6: Run forward model**
```bash
python rt_filament_forward_model.py
# Runs A-D pipeline with placeholder mu_IR=1.0
# Output: results/pipeline_results_dev.json
```

**Step 7: Run full validation**
```bash
python run_full_validation.py
# Executes complete freeze-then-test protocol
# Expected output: INCONCLUSIVE (ESG instability with 3-filament training set)
```

### 7.3 Expected results

Running the full pipeline on the v4 roster with KiDS-1000 data should reproduce:

| Metric | Expected value |
|--------|---------------|
| Tier A candidates | 19 |
| Tier A+B candidates | 28 |
| Sources per filament | 75,000 - 271,000 |
| Per-filament S/N | 3.4 - 6.4 |
| Stacked S/N (11 filaments) | ~3.9 |
| Stacked mean gamma_t | ~-0.0003 |
| mu_IR anchoring verdict | ESG FAILED |
| Overall verdict | INCONCLUSIVE |

---

## 8. Data Format Reference

### 8.1 filament_candidates.json

```json
{
  "metadata": { ... },
  "filaments": [
    {
      "filament_id": 84,           // Tempel+2014 catalog ID
      "tier": "A_primary",         // Selection tier (A/B/C/D)
      "length_mpc_h": 27.14,       // Bisous filament length (Mpc/h)
      "ngal_endpoint1": 21,        // Galaxies at endpoint 1
      "ngal_endpoint2": 27,        // Galaxies at endpoint 2
      "richness_sum": 48,          // ngal1 + ngal2
      "redshift_mean": 0.1074,     // Mean member galaxy redshift
      "endpoint1_ra": 133.1915,    // Endpoint 1 RA (degrees, J2000)
      "endpoint1_dec": -0.0554,    // Endpoint 1 Dec (degrees, J2000)
      "endpoint2_ra": 135.9737,    // Endpoint 2 RA (degrees, J2000)
      "endpoint2_dec": 2.2188,     // Endpoint 2 Dec (degrees, J2000)
      "center_ra": 134.5826,       // Midpoint RA
      "center_dec": 1.0817,        // Midpoint Dec
      "angular_sep_deg": 3.593,    // Angular extent (degrees)
      "n_member_galaxies": 90,     // Total member galaxies
      "lum_endpoint1": 28.07,      // Luminosity at endpoint 1 (10^10 L_sun)
      "lum_endpoint2": 45.76,      // Luminosity at endpoint 2 (10^10 L_sun)
      "pass_L_ge_15": true,        // Passes length cut
      "pass_z_01_02": true,        // Passes redshift cut
      "pass_ang_sep_05": true,     // Passes angular separation cut
      "pass_ngal_ge_15": true,     // Passes strict richness cut
      "pass_ngal_ge_10": true,     // Passes relaxed richness cut
      "pass_kids_footprint": true  // Both endpoints in KiDS-North
    },
    ...
  ]
}
```

### 8.2 observed_shear_profiles.json

```json
{
  "profiles": {
    "84": {
      "r_perp_bins": [0.125, 0.375, ...],  // 20 bin centers (Mpc/h)
      "gamma_t": [-0.001, 0.002, ...],     // Tangential shear per bin
      "gamma_t_err": [0.003, 0.003, ...]   // Error per bin
    },
    ...
  },
  "diagnostics": {
    "84": {
      "status": "OK",
      "n_sources_total": 224531,
      "mean_gamma_t": 0.001453,
      "mean_gamma_x": -0.000556
    },
    ...
  }
}
```

### 8.3 filament_roster_v4.json

The frozen roster used for validation. Contains 3 training + 8 holdout filaments
(all Tier A), ranked by endpoint richness sum. The training/holdout split was
determined before any WL data were examined.

---

## 9. Known Limitations

1. **Sample size**: Only 19 Tier A filaments in KiDS-North, of which 11 have
   extracted shear profiles. The filament lensing literature requires ~5000
   filaments for a robust detection (de Graaff+ 2019).

2. **Per-filament S/N**: Individual filament shear signals are at the ~10^-3 level
   with S/N of 3-6, insufficient for stable single-parameter anchoring with 3
   training filaments.

3. **KiDS footprint**: The KiDS-1000 North equatorial strip (Dec -3 to +2.5) covers
   only ~13% of the SDSS DR8 spectroscopic footprint, severely limiting the
   overlap sample.

4. **Endpoint localization**: Filament endpoints are derived from galaxy positions
   (mean of 20% nearest galaxies), not from the Bisous spine points directly.
   This introduces ~0.1-0.5 deg uncertainty in endpoint positions.

5. **Photometric redshifts**: Background source selection relies on KiDS BPZ
   photo-z estimates (Z_B), which have scatter sigma_z ~ 0.05(1+z). The
   delta_z = 0.1 buffer mitigates but does not eliminate lens-source mixing.

---

## 10. Citation

If you use this dataset, please cite:

- **Filament catalog**: Tempel, E., Stoica, R. S., Saar, E. 2014, MNRAS 438, 3465
- **KiDS-1000 shapes**: Giblin, B. et al. 2021, A&A 645, A105
- **KiDS-1000 photo-z**: Hildebrandt, H. et al. 2021, A&A 647, A124
- **Filament lensing method**: de Graaff, A. et al. 2019, A&A 624, A48

---

## 11. License

The filament candidate dataset (`data/filament_candidates.json`) is derived from
publicly available SDSS DR8 data products. The KiDS-1000 shape catalog is publicly
available from the KiDS survey team. Pipeline code is provided as-is for
reproducibility.
