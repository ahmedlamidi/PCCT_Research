# Test 0 — findings

**Verdict: AMBIGUOUS**, escalated to the extrapolation split as pre-registered.
The low-rank premise survives; the exploitable headroom is much smaller than the
raw numbers suggest, and a degree-3 baseline is the honest competitor.

Full detector (505 × 2063), 55 slab combos, 1,015,881 good pixels.

---

## Three corrections to the brief, all of which change the answer

**1. `air_table_*.raw` is byte-identical to `PMMA_0_AL_0/`** (verified by MD5, both
bins). That combo is self-normalising: `P ≡ 0` with zero noise. It was excluded —
left in, it anchors every per-pixel fit with a noiseless point and deflates the RMS.

**2. The noise floor is object-only.** The air term is common to all 56 combos, so
the per-pixel intercept absorbs it exactly, and for held-out combos the leverage
weights sum to 1 so it cancels identically. No air term enters the residual. The
model used is `Var(v) = v/600 + q/12` (600-frame average, uint16 quantisation),
with `q=2` for Low since it is a difference of two quantised bins. Quantisation is
not negligible: at the thickest combos it is ~55% of the photon variance.

**3. D4 as specified is confounded.** High is a *subset* of Total, so
`Cov(T,H) = Var(H)` and `corr(T,H) = sqrt(H/T) ≈ 0.84–0.89` under pure counting
statistics. Measured `r(Total,High)` is 0.38–0.79 — *below* its own noise floor, so
it carries no information. The informative pair is **Low vs High**: disjoint count
sets, independent under the null. This is the same `rho_LH` logic as the kill test.

---

## D1 — residual vs noise floor

`z` is the whitened residual RMS; `z = 1.00` means exactly at the photon-noise floor.

| split | deg | bin | z (raw) | z (spatial) | offset share |
|---|---|---|---|---|---|
| random_seed0 | 2 | total | 9.79 | **1.78** | 96.3% |
| random_seed0 | 3 | total | 3.92 | **1.37** | 86.7% |
| extrapolate_thick | 2 | total | 16.13 | **2.41** | 97.8% |
| extrapolate_thick | 3 | total | 2.75 | **1.38** | 78.6% |
| leave_AL_5 | 2 | total | 14.99 | **2.09** | 98.2% |
| leave_AL_5 | 3 | total | 7.35 | **1.52** | 96.2% |

**66–98% of the residual variance is a single global offset per combo** — one flat
number per acquisition, no spatial structure. The 56 slabs are separate
step-and-shoot acquisitions (`nViewTotal=1`, 600 frames each), the offsets are
~1% in log units, essentially random across combos (a degree-2 fit in thickness
gives R²≈0.43 using 6 parameters on 12 points, i.e. nothing), and they are
0.99-correlated between Total and Low. That is **tube-output drift between
acquisitions**, not detector physics. It is spatially flat, so it gives a low-rank
spatial method nothing to work with, and per-view flux normalisation removes it.

The `z (spatial)` column is the defensible D1. It never reaches 3.

## D2 — SVD of the residual (whitened)

Residuals are whitened by their predicted noise std before the SVD, so the null is
exactly iid; without this the heteroscedasticity between thin and thick combos
inflates the null's own top-5 to 0.84 and the test is meaningless.

| split | deg | top-5 energy | null | eff. rank | null rank |
|---|---|---|---|---|---|
| random_seed0 | 2 | 0.784 | 0.418 | 4.1 | 12.0 |
| random_seed0 | 3 | 0.654 | 0.418 | 7.2 | 12.0 |
| extrapolate_thick | 2 | **0.957** | 0.180 | **1.49** | 28.0 |
| extrapolate_thick | 3 | **0.956** | 0.180 | **2.07** | 28.0 |
| leave_AL_5 | 3 | 0.964 | 0.715 | 1.83 | 7.0 |

**This is the result that matters.** On extrapolation the residual is rank ~1.5–2
against a null rank of 28, and **that low-rank character is not absorbed by degree 3**
(0.957 → 0.956). The amplitude shrinks; the structure does not.

## D3 — spatial structure

The residual maps are not noise. Three superimposed patterns, all visible directly:

- a smooth **centre-to-edge gradient**, the dominant component — consistent with
  fan-angle obliquity interacting with beam hardening, which a per-pixel polynomial
  in *nominal* thickness cannot represent;
- **16 tile blocks at a 129-channel period**, derived from the data (dead columns at
  127-129, 256-258, … 1933-1935, spacing exactly 129);
- a sharp **discontinuity at row 252**, the detector midline — two stacked readout halves.

A 32-block module model (16 tiles × 2 halves) explains 5–32% of the spatial residual
variance, and its share *rises* at degree 3 (6.6% → 32.0% on extrapolation), i.e.
module structure is exactly the part a higher polynomial cannot absorb.

## D4 — cross-bin (corrected)

| split | r(Low,High) | r(Total,High) | noise-only expectation |
|---|---|---|---|
| random_seed0 deg2 | +0.033 | +0.679 | 0.837 |
| extrapolate_thick deg2 | +0.064 | +0.384 | 0.866 |
| leave_AL_5 deg2 | +0.173 | +0.596 | 0.886 |

`r(Low,High)` is +0.03 to +0.17 — essentially nothing. **The leftover error is
spatial, not spectral.** No charge-sharing or spectral-tailing signature.

## D5 — residual vs thickness

Residual RMS grows monotonically with both thicknesses and is far larger on the
thick half, which is why the extrapolation split fails hardest. At degree 2 the
trend is strong; degree 3 flattens most of it.

---

## Which branch fired

Read **literally on the raw residuals**, the rule fires **alive**: D1 > 3, D2 top-5
> 60%, not absorbed by degree 3. But it fires on tube drift — an artifact of the
slab protocol that is spatially flat and irrelevant to a spatial sharing model.
The pre-registration did not anticipate a per-acquisition offset.

Read on the **spatial** residual, which is what the project's claim is about:

- D1 max is 2.48, never > 3 → **alive fails**
- D2 is emphatically not flat (rank 1.5–2 vs 28) → **dead fails**
- → **AMBIGUOUS**

## What this means

The premise has a foundation: there is a real, spatially coherent, genuinely
low-rank residual, and it survives degree 3. Its physical identity is legible —
module boundaries plus an obliquity/hardening gradient.

But size the prize honestly. At degree 2 the spatial residual is 2.41× the noise
floor, implying a 58% ceiling on RMS reduction. **Most of that is reachable by
simply fitting degree 3**, which takes it to 1.38× — leaving a ceiling of ~27%.
The incumbent's dominant weakness is polynomial order, not un-modelable physics.

**Recommendation:** do not report the raw D1 numbers as evidence — they are drift.
Re-baseline the incumbent at degree 3, then ask whether ~27% residual reduction on
extrapolated thicknesses justifies the method. Since the surviving structure is
module-organised, a 32-parameter per-module correction is the cheap competitor that
should be beaten before a general low-rank model is worth building.

## Caveats

- Single detector, single kVp, one slab set; no repeat acquisitions, so the tube
  drift could not be measured independently and had to be inferred.
- The degree-3 comparison is bounded by the 7×8 thickness grid; degree 3 has 10
  parameters against 55 combos.
- The module model uses a boundary geometry inferred from the data, not from vendor
  documentation.
