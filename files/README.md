# Kill-test scripts

Measures the spatial-spectral noise covariance of the walnut PCCT detector and
returns GO / KILL / AMBIGUOUS. CPU only, no GPU, no training.

## Run

```bash
# 1. analysis  (~15-25 min, dominated by reading 2880 files)
python kill_test.py \
    --data /path/to/CalibrationPhantomData \
    --cal  /path/to/CalibrationTable \
    --phantom Water_Phantom \
    --out results_water

# 2. figures
python plot_results.py --results results_water --out figs_water

# 3. repeat on the second phantom — this is the transfer test
python kill_test.py --data ... --phantom HAP_Phantom --out results_hap
python plot_results.py --results results_hap --out figs_hap
```

`--data` is the directory that *contains* `Water_Phantom/` and `HAP_Phantom/`.
Needs numpy, scipy, matplotlib. ~2 GB RAM.

## Output

| File | Contents |
|---|---|
| `results.json` | every statistic, per ROI |
| `sigma_<roi>.npz` | the 9×9×3 correlation maps, per-pixel Fano and rho_LH, the good-pixel mask, count rate |
| `fig1_neighbourhood_maps.png` | 5×5 correlation structure per ROI — **this is Figure 2 of the paper** |
| `fig2_lag_decay.png` | correlation vs distance, split by direction — the decisive plot |
| `fig3_flux_trends.png` | every statistic vs count rate — checkpoint 2 |
| `fig4_summary.png` | one-panel verdict |

The `sigma_*.npz` files are the asset that feeds everything downstream. Keep them.

## What it does, in order

1. Loads the eight ROIs in one pass per bin (only ROI pixels held in RAM).
2. Builds the bad-pixel mask: supplied table + the five stuck columns
   (127, 1804–1806, 1935) + the authors' <5 / >4090 criterion.
3. Drift-normalises against an air ROI. Each air ROI is referenced against the
   *other* one, so no region is ever normalised by its own mean — that would
   artificially anticorrelate its own pixels.
4. **Harmonic-detrends the object ROIs.** The phantom is ~1.2 mm off the
   rotation axis, so its shadow slides ±27.5 channels across the detector as the
   gantry turns. Uncorrected, this geometric swing is ~6× the photon noise and
   produces a confident false GO. Air ROIs are skipped — open-beam pixels never
   see the phantom and are immune.
5. Computes Fano, the non-trivial cross-bin term, and lag-1..4 correlations in
   both directions for TT, HH and TH.
6. Prints a verdict.

## Reading the output

**`TH_x0` is excluded everywhere.** High is a subset of Total, so their
same-pixel correlation is trivially large and carries no physics. `rho_LH` is
the cross-bin number that means something — it is `Cov(T,H) − Var(H)`
normalised, and the counter arithmetic cancels out of it exactly.

**Air first.** Air ROIs need no detrending and are the cleanest evidence in the
dataset. If air and the detrended object ROIs disagree, suspect the detrending,
not the physics. The script warns when this happens.

**x vs y asymmetry is a red flag.** The wobble is purely horizontal, so residual
geometry leaks into x only. Real charge sharing is near-isotropic. Fig 2 splits
the directions so you can check this directly.

**Watch `var_removed`.** Printed per ROI. Near the shadow edge it will be
90%+ — that is the wobble, and it is expected. If it is high in `deep_centre`
too, something other than the wobble is moving.

## Validation

`make_synthetic.py` generates a fake dataset with the real geometry (off-axis
wobble, drift, fixed-pattern gain, stuck columns) and a *known* amount of
injected charge sharing, so the analysis can be checked before the real data
lands.

```bash
SHARE_FRAC=0.20 python make_synthetic.py synth 400      # correlation present
SHARE_FRAC=0.0  python make_synthetic.py synth_null 400 # nothing present
NVIEWS=400 python kill_test.py --data synth      --out res_shared
NVIEWS=400 python kill_test.py --data synth_null --out res_null
```

Results in `validation/` — 400 views, 20% horizontal sharing injected:

| | injected | null |
|---|---|---|
| peak correlation, air | **0.105** | 0.032 |
| peak correlation, object | **0.130** | 0.009 |
| nearest-neighbour TT | +0.105 | +0.001 |
| nearest-neighbour HH | +0.019 | +0.027 |
| Fano (Total) | 0.948 | 0.999 |
| verdict | **GO** | **KILL** |

Three things this confirms:

- The injected correlation is recovered, and the null case comes back flat —
  no false positive.
- The neighbourhood map reproduces the *structure* that was injected: the
  sharing was horizontal-only, and the map shows two hot pixels either side of
  centre with nothing above or below.
- Detrending works. `edge_L` had 95.5% of its variance removed as wobble, and
  still returned a clean correlation afterwards.

One caveat on the synthetic: its charge-sharing model produces near-zero
`rho_LH` by construction (the shared counts are drawn proportionally from High,
so the demotion and the spill roughly cancel in the cross-bin term). So the
validation exercises the *spatial* machinery thoroughly and the cross-bin term
only weakly. Real charge sharing should show a nonzero `rho_LH`; the synthetic
cannot confirm the script would catch it.

## Thresholds

Set at the top of `kill_test.py`:

```python
GO_THRESHOLD   = 0.10
KILL_THRESHOLD = 0.05
MAX_LAG        = 4    # 5x5 neighbourhood
N_HARMONICS    = 2    # raise if residual x/y asymmetry appears
```

Per-pair standard error is ~1/√1440 ≈ 0.026, and medians are taken over
thousands of pairs, so a null result here is a real null rather than an
underpowered one.

Ambiguous (0.05–0.10) is not a coin flip — it routes to Step 9 of the protocol:
propagate Σ through FDK and the 2×2 inverse and compare predicted material-image
noise against the diagonal-Σ prediction.
