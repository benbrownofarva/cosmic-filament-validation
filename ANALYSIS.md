# Texture Coupling Constant g_T: Measurement from Cosmic Filament Weak Lensing

## Executive Summary

We present two analyses of the RT texture coupling constant g_T using
weak-lensing tangential shear profiles measured around 11 cosmic filaments
from SDSS DR8, observed with KiDS-1000:

| Analysis | Result |
|----------|--------|
| **Matched filter extraction** | g_T = 0.54 +/- 0.33 (1.7-sigma from zero) |
| **Zero-parameter prediction** | g_T = 0.202 gives chi^2/dof = 1.02 |
| **Theory value** (Particle Sector P49) | g_T = 0.20226 |
| **Tension** | 1.05 sigma (consistent) |

The Particle Sector prediction g_T = 0.202 is consistent with the data at
1.05 sigma. The data are not yet precise enough to confirm or rule out
the predicted value — but they do not contradict it.

---

## 1. What is g_T?

The texture coupling constant g_T controls the strength of the second-Jet
anisotropy channel in Resonance Theory's emergent texture gravity framework.
It determines how strongly the deviatoric (shear) component of the
constitutive strain field couples to the observable lensing response.

In the forward model chain:

```
baryonic geometry
    --> nonlocal closure solve for kappa* (co-metric field)
    --> log-strain decomposition: theta (bulk) + eps^dev (deviatoric)
    --> STF anisotropy field Q(x; ell) at readout scale ell
    --> transverse projection Q_perp(r_perp) along filament
    --> predicted tangential shear: gamma_t = g_T * b_shear * Q_perp
```

The key property: **g_T enters as a pure overall amplitude scaling** of the
predicted lensing signal. The *shape* of the transverse profile Q_perp(r_perp)
is entirely determined by the baryonic geometry and the constitutive parameters
(lambda_IR, alpha, ell), which are fixed. Only the *amplitude* depends on g_T.

This separation of shape from amplitude is what enables us to extract g_T
without a multi-parameter fit.

### 1.1 Theoretical prediction

From the Particle Sector analysis (P49 branch-local estimate, reported in
gt_texture v0.9.0):

**g_T = 0.20226** (provisional, branch-local)

This value emerges from the micro-to-gravity pushforward (gt_texture Thm 6.1)
applied to the second-Jet activation theorem (gt_texture Thm 3.2). It is not
a free parameter — it is derived from the constitutive dynamics of the theory.

The value is classified as Tier B (provisional) in the gt_texture v0.9.0
claim-status map because it depends on the specific branch-local structure
of the particle sector. Universality across branches is conjectured but not
yet proven.

---

## 2. Data

### 2.1 Filament sample

11 cosmic filaments from the Tempel+ (2014) SDSS DR8 Bisous catalog,
selected with:
- Length >= 15 Mpc/h
- Endpoint richness >= 15 galaxies (both endpoints)
- Redshift 0.1 <= z <= 0.2
- Both endpoints within KiDS-1000 North footprint (RA 130-235, Dec -3 to +2.5)

Split: 3 training + 8 holdout (ranked by richness sum, split before WL data
examined).

### 2.2 Weak-lensing observations

Tangential shear profiles gamma_t(r_perp) extracted from the KiDS-1000 DR4.1
gold catalog (21.3 million sources, 16.5 GB):

| Filament | z | N_background | mean(gamma_t) | S/N |
|----------|---|-------------|----------------|-----|
| 84 | 0.107 | 224,531 | +0.00145 | 6.4 |
| 282 | 0.110 | 244,852 | -0.00107 | 4.1 |
| 4506 | 0.113 | 75,169 | +0.00016 | 3.4 |
| 4764 | 0.127 | 147,641 | +0.00047 | 3.5 |
| 86 | 0.110 | 271,462 | -0.00026 | 4.3 |
| 7830 | 0.116 | 154,707 | -0.00145 | 4.4 |
| 2104 | 0.121 | 141,794 | -0.00031 | 4.9 |
| 2316 | 0.121 | 147,794 | -0.00202 | 4.7 |
| 3585 | 0.125 | 85,112 | +0.00236 | 4.2 |
| 3158 | 0.139 | 181,200 | -0.00131 | 4.9 |
| 1971 | 0.105 | 141,432 | +0.00007 | 4.4 |

Each profile has 20 bins in |r_perp| from 0 to 5 Mpc/h (bin width 0.25 Mpc/h).
Errors are shape noise sigma_e = 0.28 per component divided by sqrt(N_eff).

### 2.3 Forward model templates

For each filament, the forward model produces a predicted transverse profile
Q_perp(r_perp) at unit coupling (g_T = 1, b_shear = 1, mu_IR = 1). The
predicted profiles are purely negative (Q_perp < 0), reflecting the
compressive anisotropy along the filament bridge.

Frozen constitutive parameters:
- ell = 2.0 Mpc/h (readout scale)
- sigma_kernel = 1.414 Mpc/h (Gaussian smoothing)
- lambda_IR = 1.0, alpha = 0.01 (bulk modulus, gradient penalty)
- Grid resolution: 0.5 Mpc/h

---

## 3. Analysis 1: Zero-Parameter Prediction

### 3.1 Method

Set g_T = 0.20226 (from theory) and compute:

```
gamma_t_predicted(r_perp) = g_T * b_shear * Q_perp(r_perp)
```

with b_shear = 1.0. Compare directly to observed gamma_t with no fitting.

### 3.2 Results

| Filament | Set | chi^2/dof | Sign agreement | |pred|/|obs| |
|----------|-----|-----------|----------------|-------------|
| 84 | Train | 2.07 | 0.33 | 0.036 |
| 282 | Train | 0.81 | 0.67 | 0.054 |
| 4506 | Train | 0.57 | 0.44 | 0.045 |
| 4764 | Holdout | 0.63 | 0.56 | 0.052 |
| 86 | Holdout | 0.93 | 0.56 | 0.065 |
| 7830 | Holdout | 0.92 | 0.67 | 0.043 |
| 2104 | Holdout | 1.15 | 0.50 | 0.043 |
| 2316 | Holdout | 1.06 | 0.61 | 0.040 |
| 3585 | Holdout | 0.92 | 0.33 | 0.036 |
| 3158 | Holdout | 1.17 | 0.72 | 0.049 |
| 1971 | Holdout | 0.94 | 0.39 | 0.046 |
| **Mean** | | **1.02** | **0.53** | **0.046** |

### 3.3 Interpretation

**chi^2/dof = 1.02**: This is exactly what you expect when the signal is below
the noise floor. The predicted signal (amplitude ~10^-4) is ~20x smaller than the
per-bin noise (~2x10^-3), so chi^2 is dominated by noise, giving chi^2/dof ~ 1.0
regardless of whether the prediction is correct. The prediction is *consistent
with the data* but the data cannot distinguish it from zero.

**Sign agreement = 0.53**: Consistent with random (0.50). Again, the signal
is below per-bin noise, so the sign of gamma_t in each bin is noise-dominated.

**Amplitude ratio |pred|/|obs| = 0.046**: The predicted mean gamma_t
(~1.2 x 10^-4) is about 20x smaller than the observed scatter in mean gamma_t
(~1.2 x 10^-3). This does NOT mean the prediction is wrong — the observed
values are noise, not signal. The predicted amplitude is below the noise floor
for individual filaments.

**Conclusion**: The zero-parameter prediction with g_T = 0.202 is fully consistent
with the data. It predicts a signal that is below the per-filament noise floor,
which is exactly what is observed.

---

## 4. Analysis 2: Matched Filter Extraction

### 4.1 Method

Rather than fitting multiple parameters, we use the predicted profile shape
as a **matched filter** (optimal linear estimator) to extract the single
amplitude g_T from the data.

For each filament, the matched filter estimate is:

```
          sum_b  t_b * d_b / sigma_b^2
g_hat = --------------------------------
          sum_b  t_b^2 / sigma_b^2
```

where:
- t_b = Q_perp(r_b) is the unit-coupling template at bin b
- d_b = gamma_t(r_b) is the observed tangential shear
- sigma_b = gamma_t_err(r_b) is the per-bin error

The error on g_hat is:

```
sigma_g = 1 / sqrt(sum_b  t_b^2 / sigma_b^2)
```

This is NOT an OLS fit. It is the minimum-variance unbiased estimator for
the amplitude of a known signal shape in Gaussian noise. It uses one number
from the data (the inner product with the template) to extract one number
(g_T). The template shape is entirely determined by theory — it is not fitted.

For the global (stacked) estimate, the numerators and denominators are summed
across all filaments before dividing, which is equivalent to inverse-variance
weighting of the per-filament estimates.

### 4.2 Per-filament results

| Filament | Set | g_T_hat | +/- | S/N |
|----------|-----|---------|-----|-----|
| 84 | Train | -1.63 | 0.97 | -1.67 |
| 282 | Train | +1.25 | 0.94 | +1.33 |
| 4506 | Train | +0.13 | 1.33 | +0.09 |
| 4764 | Holdout | -0.63 | 1.25 | -0.50 |
| 86 | Holdout | +0.47 | 0.77 | +0.61 |
| 7830 | Holdout | +2.12 | 1.23 | +1.72 |
| 2104 | Holdout | +1.92 | 1.15 | +1.67 |
| 2316 | Holdout | +2.17 | 1.23 | +1.77 |
| 3585 | Holdout | -2.87 | 1.45 | -1.97 |
| 3158 | Holdout | +1.45 | 0.89 | +1.63 |
| 1971 | Holdout | +0.39 | 1.32 | +0.29 |

Per-filament S/N ranges from 0.09 to 1.97. No individual filament has a
significant detection (all |S/N| < 2). The values scatter broadly, including
negative values, consistent with noise domination.

### 4.3 Global (stacked) result

Combining all 11 filaments via the matched filter:

```
g_T = 0.544 +/- 0.325
```

- **Detection significance**: 1.67 sigma (marginal)
- **Consistency with theory**: |g_hat - 0.202| / sigma = 1.05 sigma (consistent)
- **Consistency with zero**: |g_hat - 0| / sigma = 1.67 sigma (marginal)

### 4.4 Interpretation

The matched-filter extraction gives g_T = 0.54 +/- 0.33. This is:

1. **1.05 sigma from the Particle Sector prediction** (g_T = 0.202):
   The data are fully consistent with the theoretical value.

2. **1.67 sigma from zero**: There is a marginal hint of a nonzero texture
   coupling, but it is not significant at the 2-sigma level.

3. **Positive**: The central value is positive, which is the expected sign
   for the second-Jet anisotropy channel. A negative g_T would indicate
   the data prefer anti-correlation with the predicted template shape.

The error bar (0.33) is larger than the predicted value (0.20), which means
the current data cannot distinguish g_T = 0.202 from g_T = 0 with high
confidence. This is expected: the per-filament S/N is ~0.1-2.0, and even
stacking 11 filaments gives only S/N ~ 1.7.

---

## 5. The Amplitude Chain: How g_T Produces gamma_t

### 5.1 Step-by-step

Starting from two baryonic mass concentrations (filament endpoints) at
positions (x_1, y_1) and (x_2, y_2) with masses M_1, M_2:

**Step A — Source field**: The baryonic geometry produces a quadrupolar
stress-energy source aligned with the separation axis:

```
rho_dev_ij(x) = A * (n_i n_j - delta_ij/2) * corridor(x)
```

where n = (x_2 - x_1)/|x_2 - x_1| is the unit separation vector and
corridor(x) is a Gaussian profile extended along the spine.

**Step B — Screened Poisson solve**: The deviatoric strain field eps_dev is
obtained by solving:

```
(alpha * k^2 + 2 * mu_IR) * hat{eps}_dev = hat{rho}_dev
```

in Fourier space. At fixed geometry, eps_dev scales inversely with mu_IR
for large mu_IR and approaches rho_dev/alpha*k^2 for small mu_IR.

**Step C — Readout smoothing**: The STF anisotropy tensor Q is obtained by
Gaussian smoothing eps_dev at readout scale ell:

```
Q_ij(x; ell) = G_ell * eps_dev_ij
```

**Step D — Transverse projection**: Q is projected perpendicular to the
filament spine to give Q_perp(r_perp).

**Step E — Lensing observable**: The predicted tangential shear is:

```
gamma_t(r_perp) = g_T * b_shear * Q_perp(r_perp)
```

where b_shear = 1 is the deviatoric strain-optic coefficient.

### 5.2 Why g_T is a pure amplitude

In the small-strain regime, the entire chain from source to Q_perp is linear.
The profile shape Q_perp(r_perp) is determined by:
- Endpoint positions and masses (geometry)
- Constitutive parameters lambda_IR, alpha (PDE structure)
- Readout scale ell and kernel sigma (smoothing)

All of these are fixed before g_T enters. The coupling constant g_T multiplies
the final result as an overall amplitude. This means:

1. We can compute Q_perp once at unit coupling
2. The predicted gamma_t at any g_T is just gamma_t = g_T * Q_perp
3. The matched filter directly extracts g_T without multi-parameter degeneracies

### 5.3 Numerical values

At unit coupling (g_T = 1, mu_IR = 1), the predicted profiles have:
- Peak |Q_perp| ~ 10^-3 (in the inner 1-2 Mpc/h)
- Integrated amplitude A_fil ~ -6 x 10^-3

At g_T = 0.202:
- Peak |gamma_t| ~ 2 x 10^-4
- Mean |gamma_t| ~ 1.2 x 10^-4

The observed per-bin errors are ~2-3 x 10^-3, so the predicted signal at
g_T = 0.202 is about 10-20x below the per-bin noise floor.

---

## 6. Statistical Framework

### 6.1 Why not OLS?

Ordinary least-squares fitting of g_T to the 20 x 11 = 220 data points
would formally work, but it:

1. Treats the problem as if there are 220 independent constraints on 1
   parameter, when in reality the constraining power comes from the
   coherent template shape
2. Is sensitive to correlated noise across bins (which we have not modeled)
3. Gives exactly the same point estimate as the matched filter, but
   obscures the simplicity of the extraction

The matched filter makes transparent that we are extracting *one number*
(the amplitude) from *one template* — not fitting a curve.

### 6.2 Error budget

The dominant error source is **shape noise**: the intrinsic ellipticity
dispersion of background galaxies (sigma_e = 0.28 per component for
KiDS-1000). This produces ~2-3 x 10^-3 per-bin errors for our filament
corridors (75K-271K sources each).

Subdominant contributions (not formally propagated):
- Photo-z scatter: delta_z ~ 0.05(1+z), mitigated by 0.1 buffer
- Endpoint localization: ~0.1-0.5 deg uncertainty
- Constitutive parameter uncertainty: lambda_IR, alpha not independently constrained
- Intrinsic alignments: not modeled (potentially ~10% of signal at these scales)

### 6.3 What would it take to reach 5-sigma?

The current stacked S/N is 1.67 with 11 filaments. Since S/N scales as
sqrt(N_filaments):

```
N_needed = 11 * (5 / 1.67)^2 ~ 99 filaments
```

This requires either:
- Relaxing selection cuts (we have 558 candidates at L>15, ngal>10)
- Using a wider WL survey (Euclid: ~15,000 deg^2 vs KiDS ~1,350 deg^2)
- Or both: Euclid + SDSS would provide ~1000+ filaments in overlap

---

## 7. Conclusion

The RT texture coupling constant g_T = 0.20226, predicted from the
Particle Sector constitutive dynamics, is **consistent with KiDS-1000
weak-lensing observations of 11 SDSS cosmic filaments** at the 1.05-sigma
level.

The matched-filter extraction gives g_T = 0.54 +/- 0.33:
- Positive (correct sign for second-Jet activation)
- 1.67 sigma from zero (marginal hint of nonzero texture coupling)
- 1.05 sigma from the predicted value (no tension)

The current dataset cannot distinguish g_T = 0.202 from g_T = 0 at high
significance. This is a **statistical power limitation**, not a falsification.
Approximately 100 filaments in a WL survey overlap would be needed for a
5-sigma measurement.

### What this result does and does not say

**Does say:**
- The data do not contradict g_T = 0.202
- The predicted signal amplitude is below the per-filament noise floor, exactly as observed
- The matched filter finds a positive central value (correct sign)
- The pipeline works end-to-end on real data

**Does not say:**
- g_T = 0.202 is confirmed (insufficient statistical power)
- The second-Jet channel is detected (only 1.67 sigma)
- Other values of g_T are ruled out (the error bar is wide)

---

## 8. Reproducibility

All results can be reproduced with:

```bash
# 1. Install dependencies
pip install numpy scipy astropy requests

# 2. Download data
python data/download_sdss_filaments.py
python data/download_kids_gold.py  # 16.5 GB

# 3. Run analysis
cd pipeline
python g_T_analysis.py
```

The analysis script `pipeline/g_T_analysis.py` reads the observed shear
profiles and forward model templates, runs both the zero-parameter prediction
and matched filter extraction, and saves results to `results/g_T_analysis.json`.

Input data hashes:
- Roster v4: SHA-256 `25c77897b5428adc...`
- SDSS DR8 catalog: `dr8_filaments.fits` (52.8 MB, 15,421 filaments)
- KiDS-1000 gold: `KiDS_DR4.1_WL_gold_cat.fits` (16.5 GB, 21.3M sources)
