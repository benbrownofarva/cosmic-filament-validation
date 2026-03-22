# SDSS Cosmic Filament Weak-Lensing Dataset

Curated cosmic filament sample from SDSS DR8 with KiDS-1000 tangential shear profiles.

| Item | Value |
|------|-------|
| Source catalog | Tempel, Stoica & Saar 2014, MNRAS 438, 3465 |
| WL survey | KiDS-1000 DR4.1 (Giblin+ 2021) |
| Candidates (all tiers) | 558 |
| With KiDS-1000 shear profiles | 11 |

---

## Data Products

### `data/filament_candidates.json` and `.csv`

558 filament candidates passing base selection (L >= 15 Mpc/h, 0.1 <= z <= 0.2,
ngal >= 10 both endpoints, angular separation >= 0.5 deg), with per-filament
boolean flags for each criterion and a tier label:

| Tier | Criteria | Count |
|------|----------|-------|
| A_primary | ngal >= 15, in KiDS-North | 19 |
| B_relaxed | ngal >= 10 (not 15), in KiDS-North | 9 |
| C_allsky | ngal >= 15, outside KiDS | 335 |
| D_allsky_relaxed | ngal >= 10, outside KiDS | 195 |

Each record contains:

| Field | Description |
|-------|-------------|
| `filament_id` | Tempel+ 2014 catalog ID |
| `tier` | Selection tier (A/B/C/D) |
| `length_mpc_h` | Bisous filament length (Mpc/h) |
| `ngal_endpoint1`, `ngal_endpoint2` | Galaxies at each endpoint |
| `redshift_mean` | Mean member galaxy redshift |
| `endpoint1_ra`, `endpoint1_dec` | Endpoint 1 position (J2000 deg) |
| `endpoint2_ra`, `endpoint2_dec` | Endpoint 2 position (J2000 deg) |
| `angular_sep_deg` | Angular extent (deg) |
| `n_member_galaxies` | Total member galaxies |
| `lum_endpoint1`, `lum_endpoint2` | Endpoint luminosity (10^10 L_sun) |
| `pass_L_ge_15` | Passes length cut |
| `pass_z_01_02` | Passes redshift cut |
| `pass_ang_sep_05` | Passes angular separation cut |
| `pass_ngal_ge_15` | Passes strict richness cut |
| `pass_ngal_ge_10` | Passes relaxed richness cut |
| `pass_kids_footprint` | Both endpoints in KiDS-North |

### `results/observed_shear_profiles.json`

KiDS-1000 tangential shear profiles gamma_t(r_perp) for 11 filaments (the 19
Tier A candidates were split 3 + 8 before WL data were examined; all 11 yielded
usable profiles).

Per filament: 20 bins in |r_perp| from 0 to 5 Mpc/h. Each bin has `gamma_t`,
`gamma_t_err`, and source counts. Diagnostics include cross-component gamma_x
(B-mode null).

| Filament | z | N_background | mean(gamma_t) |
|----------|---|-------------|----------------|
| 84 | 0.107 | 224,531 | +0.00145 |
| 282 | 0.110 | 244,852 | -0.00107 |
| 4506 | 0.113 | 75,169 | +0.00016 |
| 4764 | 0.127 | 147,641 | +0.00047 |
| 86 | 0.110 | 271,462 | -0.00026 |
| 7830 | 0.116 | 154,707 | -0.00145 |
| 2104 | 0.121 | 141,794 | -0.00031 |
| 2316 | 0.121 | 147,794 | -0.00202 |
| 3585 | 0.125 | 85,112 | +0.00236 |
| 3158 | 0.139 | 181,200 | -0.00131 |
| 1971 | 0.105 | 141,432 | +0.00007 |

Stacked (inverse-variance weighted): mean gamma_t = -0.0003 +/- 0.0010, S/N ~ 3.9.

### `rosters/filament_roster_v4.json`

The frozen 3 training + 8 holdout split used for shear extraction. Training/holdout
assignment was made by richness ranking before any WL data were examined.

---

## Selection Criteria

### Source catalog

15,421 filaments from the Tempel, Stoica & Saar (2014) Bisous catalog applied to
SDSS DR8. Downloaded from https://www.aai.ee/~elmo/sdss-filaments/dr8_filaments.fits
(52.8 MB FITS, 3 HDUs: FILAMENTS, FILPOINTS, GALAXIES).

### Selection funnel

| Step | Cut | Remaining |
|------|-----|-----------|
| 0 | All filaments with galaxies | 15,421 |
| 1 | Length >= 15 Mpc/h | 2,306 |
| 2 | Redshift 0.1 <= z <= 0.2 | 600 |
| 3 | Angular separation >= 0.5 deg | 599 |
| 4a | ngal >= 15 both endpoints | 354 |
| 4b | ngal >= 10 both endpoints | 558 |
| 5 | KiDS-North footprint | 19 (4a) / 28 (4b) |

### Cut rationale

- **L >= 15 Mpc/h**: Ensures >= 3 resolution elements at the 5 Mpc/h corridor width.
- **0.1 <= z <= 0.2**: SDSS spectroscopic completeness; sufficient angular diameter
  distance for resolved transverse profiles.
- **Angular sep >= 0.5 deg**: Excludes artefacts/compact structures.
- **ngal >= 15 (Tier A)** / **>= 10 (Tier B)**: Robust endpoint localization.
- **KiDS-North**: RA [130, 235], Dec [-3, +2.5] — verified empirically from the
  21.3M-source KiDS-1000 DR4.1 gold catalog.

### Endpoint extraction

Endpoints are not stored in the Bisous catalog directly. For each filament:
1. Select member galaxies (`fil_id == filament_id`)
2. Sort by `fil_idpts` (spine-point index, global into FILPOINTS table)
3. Endpoint 1 = mean RA/Dec of first 20% of sorted galaxies (min 5)
4. Endpoint 2 = mean RA/Dec of last 20% (min 5)

---

## Shear Extraction Method

For each filament, gamma_t(r_perp) is extracted from the KiDS-1000 DR4.1 gold
WL catalog (21,262,011 sources):

1. **Background cut**: Z_B > z_filament + 0.1
2. **Quality cuts**: fitclass == 0, weight > 0, Z_B < 1.2, finite e1/e2
3. **Corridor**: rectangular region, width = 10 Mpc/h, length = filament + 4 Mpc/h buffer
4. **Tangential shear**: gamma_t = -(e1 cos 2phi + e2 sin 2phi)
5. **Binning**: 20 bins in |r_perp|, 0 to 5 Mpc/h
6. **Weighting**: lensfit weight * Sigma_crit^{-1}(z_l, z_s)
7. **Errors**: sigma_e / sqrt(N_eff), sigma_e = 0.28

Code: `pipeline/extract_wl_shear.py`

---

## Reproducing

```bash
pip install numpy scipy astropy requests

# Download source data
python data/download_sdss_filaments.py          # 53 MB
python data/download_kids_gold.py               # 16.5 GB (or wget directly)

# Extract shear profiles
cd pipeline && python extract_wl_shear.py
```

---

## Citation

- Tempel, E., Stoica, R. S., Saar, E. 2014, MNRAS 438, 3465
- Giblin, B. et al. 2021, A&A 645, A105
- Hildebrandt, H. et al. 2021, A&A 647, A124
- de Graaff, A. et al. 2019, A&A 624, A48
