# Week-1 Kill Test — Correlated-Noise-Aware Spectral Decomposition

**Purpose:** decide, before writing any method code, whether the walnut PCCT dataset contains measurable spatial-spectral noise correlation. If it doesn't, the project dies here and we fall back to Gap #2 (flux-conditioned learned inverse).

**Cost if it fails:** a ~9 GB download and a few days.

---

## The claim being tested

Deep learning material decomposition uses noise-blind losses (MSE and friends), which implicitly assume noise is independent across pixels and bins. In a photon-counting detector it isn't: a photon landing near a pixel border splits its charge between neighbours, producing linked errors in two pixels and two bins from one physical event. The proposed contribution is a training objective built on a *measured* spatial-spectral noise covariance.

The 2×2 decomposition (Total ≥15 keV, High ≥30 keV, Low by subtraction) is the worst-conditioned case, which maximises the payoff from getting the covariance right.

This test asks one question: **is the correlation large enough to build on?**

---

## What the paper is actually about

Framing matters here, and it changes what the kill test needs to collect.

A whitened residual is a Gaussian likelihood with a full covariance — textbook. If the thesis is "I used a better loss," a reviewer writes *this is generalized least squares* and the submission is in trouble. The contribution has to be the **noise model** and what follows from it, not the algebra.

Thesis: **the community's noise model for photon-counting material decomposition is measurably wrong at the second moment, here is the corrected model, and here is what correcting it buys.**

Four checkpoints, three of which are measurements or derivations rather than training results — so the paper survives even if the network gain is modest:

1. **Measure** the spatial-spectral covariance Σ from real PCD data.
2. **Parameterise** it as a function of local flux, Σ(φ), rather than one fixed matrix.
3. **Propagate** it analytically through FDK and the 2×2 inverse to predict material-image noise — then verify the prediction against observation.
4. **Apply** it as a training objective and measure the gain.

Both 2 and 3 need data this kill test can collect at essentially no extra cost, which is why the ROI design below is six flux-binned populations rather than two.

---

## Prior art position (revised after literature sweep, Aug 2026)

The gap survives but is narrower than the first sweep suggested. Each checkpoint below is annotated with what is already published.

### Checkpoint 1 — measurement: **occupied, use survives**

**Winfree et al., Med Phys 51 (2024), doi 10.1002/mp.17263** (Mayo, Lifeng Yu). Built a noise insertion algorithm for PCD-CT. To validate it they acquired pairs of identical water-tank scans, subtracted the projections to get noise-only data, and computed correlation between energy channels and between adjacent pixels. They note that removing inter-pixel septa raises cross-talk, inducing positive covariance between nearby pixels and violating spatial independence, while pileup induces negative covariance between bins.

That is the measurement. **"First measurement of PCD spatial-spectral covariance" is not available as a claim.** What survives: they measured it to drive a *simulator*, never touched material decomposition, and never used it in an estimation objective. Reframe as first *use*, cite prominently, and adopt their difference-of-repeated-scans method as an independent cross-check on your temporal-variance approach.

### Checkpoint 2 — flux conditioning: **open. This is now the strongest piece.**

Nothing found makes Σ a function of operating point. Roessl and Tang both work at a single flux; the MICCAI 2024 method treats physics parameters as fixed calibration constants. Winfree varied mA but for noise-equivalent photon number linearity, not for a covariance model.

This is why the six-ROI design matters and why the flux axis should be prominent in the paper rather than an appendix.

**Caveat found in the data (see Step 5).** The water phantom's flux range is only ~4.8× and is confounded with beam hardening — the High/Total ratio moves from 0.545 in air to 0.724 at the centre. Rate-dependence and spectral-dependence cannot be separated from this phantom alone. The HAP phantom provides a second operating point and partial disentangling; the limitation must be stated rather than glossed. Checkpoint 2 is still the least-occupied ground, but it is weaker than first assessed.

### Checkpoint 3 — propagation: **partly occupied, boundary not yet settled**

**Roessl, Ziegler & Proksa, Med Phys 34(3):959–966 (2007)** — Philips Hamburg. Derives how *input-side* measurement correlation propagates into basis image noise, for a counting-plus-integrating readout where both channels are electronically derived from one signal. Structurally the same situation as your Total/High nesting: correlation by construction rather than by physics. Your `Cov(T,H) − Var(H)` cancellation argument must cite this.

**Tang, Ren & Xie, J Appl Clin Med Phys 24(1):e13830 (2023)**, expanded as a 2024 Springer chapter (doi 10.1007/978-3-031-63897-8_6). Derives equations for noise and noise correlation in material-specific images for m-material decomposition, verified in simulation on a digital phantom and modified Shepp–Logan. Key result: in two-material decomposition the correlation coefficient between the two basis images approaches −1; for m ≥ 3 pairwise correlations alternate between ±1; VMI noise stays moderate even when basis noise varies drastically. Companion: Tang & Ren, Med Phys 48(3):1100–1116 (2021), on conditioning of basis materials.

**Unresolved — do this before writing method text.** Full text was inaccessible (PMC and Wiley both block automated access). Working read: Roessl covers *input* correlation, Tang covers correlation *between output* basis images induced by the inversion, implying Tang's input covariance is diagonal. If that's right, the per-pixel propagation of a non-diagonal input covariance is still open. If Tang admits a general input covariance, checkpoint 3 shrinks to the spatial term alone.

> **Action: read Tang 2023 §Method and answer one question — does the derivation admit a non-diagonal input covariance, or only a diagonal one?** Ten minutes, and it sets the boundary of checkpoint 3.

Either way, what survives: the **cross-pixel** term, which requires passing through reconstruction rather than only the 2×2 matrix, and **empirical validation against real data** rather than simulation. Tang's −1 becomes a result you reproduce as a sanity check, not a finding.

### Checkpoint 4 — physics in DL decomposition: **open, and the closest paper is an ally**

**Yu et al., MICCAI 2024, paper 3032** (ShanghaiTech + United Imaging). Monte Carlo full-chain PCCT model covering energy deposition, charge transport, charge sharing, ASIC dead time, pileup and trigger level. Two networks (Detector Net, ASIC Net) emulate the forward response given physics parameters; calibration fits those parameters to slab measurements by gradient descent; decomposition inverts for material depths, then FBP.

**Their loss is the maximum log-Poisson** — an independent-Poisson likelihood.

So they model charge sharing meticulously in the *first* moment and assume independence in the *second*. This is not a competitor; it is the gap demonstrated by someone else at your target venue. Frame it as: physics has been supplied to the mean, never to the covariance. Also useful practically — your diagonal-weighted ablation row is now reproducing a real published objective rather than a strawman.

### The Philips patent — tighter than first assessed

The patent (vector-valued image denoising) claims a loss function built from covariance matrices, with submatrices modified per spatial frequency band to adjust assumed inter-material correlation in the cost function. That *is* covariance-in-a-loss, so the earlier "post-decomposition, therefore distinguishable" shorthand is not enough.

The distinction, which needs a full paragraph in related work: theirs is a denoising cost applied to already-decomposed material images with a heuristic, hand-modified covariance; yours is a training objective for the decomposition itself, with a covariance measured from detector data. Note also that Roessl 2007 and this patent are both Philips — **they have worked this seam for twenty years and are the most likely group to close the gap first.**

### The surviving claim, stated precisely

> Detector-level spatial-spectral noise correlation has been modelled theoretically (Taguchi; Faby 2016; Tanguay 2020) and measured on real hardware for noise simulation (Winfree 2024). Its propagation into basis images has been derived for electronically-correlated channels (Roessl 2007) and for multi-material decomposition (Tang 2023), both per-pixel and in simulation. Detector physics has been supplied to deep-learning decomposition through the forward model and the first moment, under an independent-Poisson objective (Yu, MICCAI 2024). No work has measured the spatial-spectral covariance of a real photon-counting detector, parameterised it by flux, and used it as the training objective for material decomposition.

---

## Configuration in force

Change any of these and the affected steps change with them.

| Choice | Setting | Why |
|---|---|---|
| Data scope | Cropped ROIs, not full frames | ~190 MB vs ~6 GB; fast iteration |
| Populations | **6 ROIs binned by attenuation**, not 2 | Gives a flux axis for Σ(φ) from a single scan — see Step 5 |
| Neighbourhood | 5×5 (lags 1–4) | Gives a correlation decay length, separating charge sharing (~1 px) from K-fluorescence (further) |
| Drift reference | Air ROI mean | Air pixels never see the phantom, so they are immune to the off-axis wobble (Step 6) |
| Geometry handling | **Harmonic detrend, mandatory** | Phantom is confirmed ~1.2 mm off-axis; without this the test returns a false GO (Step 6) |
| Author email | Send now, in parallel | AcqPara has no detector-mode field, so this is the *only* route to the anti-coincidence answer |

## Resources

- **CPU only.** No network training here, just counting statistics.
- **Disk:** ~25 GB free (9.2 GB zip plus extraction).
- **RAM:** 8 GB workable with cropping and memmap; 16 GB comfortable.
- **Compute:** the 18×18 neighbourhood covariance over ~16k locations is ~7.6 GFLOP — seconds in numpy. Using `np.roll` for lag correlations makes it sub-second per lag.
- **Wall clock:** dominated by download and unzip. Analysis itself is minutes.
- **Python:** numpy, scipy, matplotlib, h5py (for v7.3 `.mat`) or scipy.io (older).

One frame = 505 × 2063 × 2 bytes = **2,083,630 bytes**. Both bins × 1440 views = ~6.0 GB as uint16, ~12 GB cast to float32 — which is why we crop.

---

## Step 0 — Download

**https://doi.org/10.5281/zenodo.17328375** — the "calibration table & sample1" bundle.

> **The DOI in the paper (10.5281/zenodo.15738313) is superseded.** It redirects to record 17328375, published 2025-10-11. Same four files. Cite the version you actually download.

- `CalibrationPhantomData.zip` — 9,230,791,362 bytes exactly
- `CalibrationTable.zip` (~62 MB) — bad-pixel index

Start this first; it's the long pole. Direct link:
`https://zenodo.org/api/records/17328375/files/CalibrationPhantomData.zip/content`

The other three files in the record (Reconstructions.zip 19 GB, Walnut_1.zip 17.9 GB) are not needed for the kill test.

Author code (load conventions, bad-pixel indexing): https://github.com/zezisme/WalnutPCCTReconCodes

### Acquisition parameters (from the data descriptor)

| | |
|---|---|
| Detector | XCounter THOR-FX20, CdTe, 100 µm pitch |
| Native / effective resolution | 2063 × 513 → 2063 × 505 (4 rows cropped top and bottom) |
| Counter depth | **12-bit, max 4096 counts per acquisition** |
| Thresholds | 15 keV (Total), 30 keV (High); Low = Total − High |
| Source | 80 kV, 200 µA, 0.5 mm Al filter |
| Exposure | 70 ms per projection, chosen to stay inside dynamic range |
| Views | 1440 at 0.25° increment, continuous mode |
| Geometry | SID 140 mm, SDD 325 mm, FOV 80 mm, gantry rotates around stationary object |
| Raw format | uint16, little-endian, Width 2063 × Height 505 |

### Verified from AcqPara.mat (extracted directly from the archive)

Confirms the descriptor and adds four fields that matter:

| Field | Value | Why it matters |
|---|---|---|
| `nFramesNumPerView` | **1** | Single shot per view — **no on-scanner averaging, noise is intact.** The premise of the whole test. |
| `U0` | **983.52** | Detector channel where the rotation axis projects. Reference for the Step 6 wobble check. |
| `V0` | **253.99** | Centre slice of 505. |
| `BinningMode` | Mode1x1 | No pixel binning. |
| `InpRot` | 0.122 | Small in-plane detector rotation. |
| `objViewCouchPosition` | constant 585 | Couch does not move; single axial rotation. |
| `objViewAngle` | 0.0025 → 6.281 rad | Full 360°, continuous. |

**There is no detector-mode, charge-summing, or anti-coincidence field.** The AcqPara route to that question is exhausted — only the author email remains (Step 10).

## Step 1 — Extract only `Water_Phantom`

Structure is confirmed: `Water_Phantom/` contains `High/` and `Total/`, each holding `AcqPara.mat` and `proj_<j>.raw` for j = 00001–01440.

```python
import zipfile
with zipfile.ZipFile('CalibrationPhantomData.zip') as z:
    names = [n for n in z.namelist() if 'Water_Phantom' in n]
    z.extractall('data/', members=names)
```

Only `Water_Phantom` is usable: 1440 raw projections of a static, rotationally symmetric cylinder, i.e. 1440 independent noise realisations of an unchanging scene. The PMMA/Al slabs were saved as 600-frame **averages**, so their noise is already destroyed.

### HAP_Phantom — confirmed per-view. Extract this too.

Verified directly from the archive's central directory:

```
HAP_Phantom/High/    1440 × proj_00001..01440.raw  + AcqPara.mat
HAP_Phantom/Total/   1440 × proj_00001..01440.raw  + AcqPara.mat
Water_Phantom/High/  1440 × proj_00001..01440.raw  + AcqPara.mat
Water_Phantom/Total/ 1440 × proj_00001..01440.raw  + AcqPara.mat
```

Every raw file is exactly 2,083,630 bytes. 5,941 entries total, 12.24 GB uncompressed.

The QRM phantom holds hydroxyapatite rods at **known densities (50, 100, 200 mg/cm³)** in a PMMA holder — the only genuine quantitative ground truth in this dataset, and HAP is a bone surrogate, which reads far more medical to an IPMI reviewer than shell-vs-pulp. **This is now the leading candidate for the primary quantitative result.**

It also sits at a different attenuation from the water phantom (single-view mean 2269 vs 1516 counts in Total), which gives a second, independent operating point — useful for the Σ(φ) transfer test that the IPMI framing depends on.

## Step 2 — Verify raw format — ~~HARD GATE~~ **already passed**

Every `.raw` entry in the archive is exactly **2,083,630 bytes** = 505 × 2063 × 2. No header. Three files were decompressed and reshaped successfully as `'<u2'` (little-endian uint16). Nothing to check.

Keep the assertion in the loader anyway as cheap insurance:

```python
assert os.path.getsize(path) == 2083630
a = np.fromfile(path, dtype='<u2').reshape(505, 2063)
```

### Measured single-view values (Water_Phantom, view 1)

| | Total bin | High bin |
|---|---|---|
| mean / median | 1516 / 915 | 917 / 641 |
| max | 4095 | 3430 |
| pixels ≥ 4095 | 2029 | 0 |
| pixels < 5 | 21 | 20 |

**Correction to earlier guidance: there is no counter saturation.** Open-beam Total sits at ~2895 counts with Poisson sigma ~54, leaving 22 sigma of headroom to the 4096 ceiling. The previous "air mean above ~3000 means truncation" heuristic was badly calibrated — truncation only matters within a few sigma of the ceiling. **The air ROI stays in the statistics.**

The pixels at 4095 are not saturation. They are **five stuck-high columns: 127, 1804, 1805, 1806, 1935**, pinned at 4095 down their full 505-row length. These should appear in the bad-pixel table; if they don't, mask them by hand.

## Step 3 — Build the two stacks

Two lists of 1440 paths each, from `Total/` and `High/`, in matching view order.

Verify both are length 1440, and that view *j* of Total corresponds to view *j* of High. An off-by-one between bins destroys the cross-bin term specifically, and does so invisibly.

Don't load yet.

## Step 4 — Bad-pixel mask

`badchannelIndexAll.data` in `CalibrationTable.zip`, stored as **float32**. Build a boolean array of shape (505, 2063), True = good.

The list includes the **physical gaps between detector tiles** as well as pixels with abnormal response. That matters directly: a neighbour pair spanning a tile gap is physically meaningless and must be excluded, not just masked.

Additionally flag any pixel whose counts fall below 5 or above 4090 in any view — that's the authors' own criterion.

Backup if the format isn't obvious: average 50 frames, flag pixels >5σ from the local median, plus any pixel with zero variance across views.

## Step 5 — Pick ROIs, binned by flux

The single change that buys checkpoint 2. A rotationally symmetric cylinder gives a natural attenuation gradient across the detector: rays through the centre traverse the full diameter, rays near the edge traverse almost nothing. So one scan already contains a flux sweep — you don't need extra data or a second operating point.

```python
avg = np.mean([np.fromfile(p, dtype='<u2').reshape(505,2063).astype(np.float32)
               for p in total_paths[:50]], axis=0)
plt.imshow(avg); plt.colorbar()
```

From the displayed image, hardcode **six ROIs of ≥96×96** spanning open beam to cylinder centre. Constraints on every ROI:

- **No stuck columns** (127, 1804, 1805, 1806, 1935) and no detector tile gaps. A neighbour pair spanning either is meaningless.
- **Not on the shadow edge.** Edge pixels move most under the off-axis wobble (Step 6) and have the steepest gradient.
- **Approximately uniform mean within the ROI.** A strong gradient inside a box mixes flux levels and defeats the binning.

Record each ROI's mean count rate. **That number is the x-axis of Σ(φ)** — without it the flux extension needs a reprocess.

### Measured profile — use these as starting coordinates

From view 1, rows 190–310 (central slices, away from the V edges):

| Region | Channels | Total | High | High/Total | Profile gradient |
|---|---|---|---|---|---|
| open beam L | 40–180 | 2895 | 1577 | 0.545 | 32.4 |
| open beam R | 1750–1900 | 2945 | 1596 | 0.542 | 30.3 |
| shadow edge L | 330–430 | 1096 | 727 | 0.663 | 20.2 |
| light attenuation | 450–570 | 816 | 568 | 0.696 | 11.6 |
| medium attenuation | 620–760 | 659 | 472 | 0.715 | 8.2 |
| deep / flat centre | 850–1150 | 599 | 433 | 0.724 | 5.6 |
| medium attenuation R | 1250–1400 | 718 | 511 | 0.712 | 8.5 |
| light attenuation R | 1480–1600 | 1063 | 706 | 0.664 | 16.0 |

The phantom shadow spans roughly channels 302–1634.

### Two honest caveats about the flux axis

**The range is narrower than hoped.** Air is 2895; everything under the phantom clusters between 599 and 1096. That's a 4.8× total range with most points bunched near 600. Thin for fitting Σ(φ), though workable — and the HAP phantom adds a second operating point at a different attenuation.

**Flux is confounded with beam hardening.** The High/Total ratio climbs from 0.545 in air to 0.724 at the centre, because water preferentially removes low-energy photons. So the "flux axis" is simultaneously a spectrum axis, and this phantom alone **cannot separate rate-dependence (pileup) from spectral-dependence**. Mitigations: use High/Total as a hardening covariate in the fit, compare against the HAP phantom at matched count rate but different spectrum, and state the limitation explicitly in the paper rather than glossing it. This is a real weakening of checkpoint 2 as originally conceived.

Charge sharing is roughly flux-independent; pileup is not. Watching each statistic move across the six bins still separates the two mechanisms, subject to the caveat above.

Everything after this reads only these boxes.

## Step 6 — Geometry correction (HARD GATE — **already failed, fix is mandatory**)

This is no longer a check. It was run on eight projections pulled directly from the archive, and **the phantom is off the rotation axis.**

### The measurement

Attenuation centroid of the water cylinder, rows 190–310, versus gantry angle:

| View | Angle | Centroid (channel) |
|---|---|---|
| 1 | 0° | 967.5 |
| 181 | 45° | 958.8 |
| 361 | 90° | 966.0 |
| 541 | 135° | 984.6 |
| 721 | 180° | 1004.3 |
| 901 | 225° | 1013.9 |
| 1081 | 270° | 1007.0 |
| 1261 | 315° | 987.5 |

Clean sinusoid — minimum at 45°, maximum at 225°, exactly 180° apart, oscillating about U0 = 983.5. Peak-to-peak **55.1 channels**, so **±27.5 channels**. Scaling by SID/SDD = 140.244/324.335 gives roughly **1.2 mm off-axis at the object**.

### Why this would have killed the test silently

In the flat centre the profile gradient is ~5.6 counts/channel. A ±27.5 channel sweep therefore swings a fixed pixel by about **±155 counts**. Poisson sigma there is **24.5 counts**.

**The geometry is ~6× larger than the noise being measured.** Naive per-pixel temporal variance across 1440 views would be dominated by the phantom sliding across the detector, producing large positive neighbour correlation with nothing to do with charge sharing — a confident **false GO**.

### The fix — harmonic detrending, mandatory for every object ROI

The motion is exactly one sinusoid per rotation, so it is removable in closed form. Per pixel, regress counts on view angle θ and keep the residual:

```python
th = view_angle                      # radians, from AcqPara objViewAngle
X  = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th),
                      np.cos(2*th), np.sin(2*th)])       # 5 params
beta = np.linalg.lstsq(X, roi.reshape(len(th), -1), rcond=None)[0]
resid = roi.reshape(len(th), -1) - X @ beta
```

Five parameters against 1440 samples — a 0.35% DOF cost, negligible. The second harmonic is insurance against a slightly elliptical or tilted phantom.

**Verification after detrending:** the residual mean must be flat in θ, and residual variance should drop to roughly the Poisson level. If variance is still several times the mean after detrending, something else is moving and you should stop and investigate rather than proceed.

### The air ROI is immune — use it as the anchor

The gantry rotates around a stationary object, so the detector-fixed pattern (bowtie, heel effect, pixel gain) never moves, and **open-beam pixels never see the phantom at all.** Air pixels are therefore genuinely static, need no detrending, and any correlation found there is real.

**Run the air population first.** It's the cleanest evidence in the dataset and carries none of this risk. If air shows correlation and the detrended object ROIs agree, the result is solid. If they disagree, the detrending is suspect.

### One more thing this reveals

Spatial variance in open beam runs about **10× Poisson** — that's the pixel and tile non-uniformity the authors built STEPC to correct. It confirms there is no spatial shortcut: every statistic must come from the temporal direction across views, never from variance within a single frame.

## Step 7 — Drift normalisation

Tube output wanders over the ~20-minute acquisition. That is a common-mode multiplicative fluctuation hitting every pixel at once, and it reads as enormous positive cross-pixel correlation having nothing to do with charge sharing.

```python
air_mean = roi_air.mean(axis=(1,2))                    # (1440,)
keep = np.abs(air_mean - np.median(air_mean)) < 5*air_mean.std()
roi_obj_n = roi_obj[keep] / air_mean[keep][:,None,None] * air_mean[keep].mean()
```

Apply the same to the air ROI, but reference it against a **disjoint** air region — normalising a region by its own mean artificially anticorrelates its pixels and corrupts the air-population result.

Record how many views `keep` dropped.

**Order matters:** drift-normalise *before* harmonic detrending. Drift is multiplicative and common-mode; the wobble is geometric and pixel-specific. Removing drift first means the harmonic fit isn't absorbing tube fluctuation into its coefficients.

Second-order caveat: this injects the reference ROI's own noise into every pixel. Negligible for a large ROI, but real.

## Step 8 — The three statistics

Run all of it on **each of the six populations**, computed across the view axis. Tabulate every result against that ROI's mean count rate — the trend across flux is as informative as any single value.

### (a) Fano factor

```python
fano = roi.var(axis=0) / roi.mean(axis=0)
```

Report median and IQR per bin. 1.0 = pure Poisson. Below 1 suggests anti-coincidence or deadtime; above 1 suggests sharing or double-counting.

### (b) Non-trivial cross-bin term

With H = counts above 30 keV, L = counts 15–30 keV, T = H + L:

If H and L were independent Poisson (no sharing), then Cov(T,H) = Var(H) + Cov(L,H) = Var(H) exactly. So **Cov(T,H) − Var(H) is identically Cov(L,H), which is zero in the ideal case.** The deterministic counter nesting cancels analytically; whatever remains is physics — negative if sharing pushes events down out of High, positive if double-counting adds to both.

```python
cov_TH = ((T - T.mean(0)) * (H - H.mean(0))).mean(0)
nontrivial = cov_TH - H.var(0)
var_L = T.var(0) + H.var(0) - 2*cov_TH
rho_LH = nontrivial / np.sqrt(var_L * H.var(0))
```

Report median and IQR of `rho_LH`. This is the number that answers the two-bin objection.

### (c) Neighbour correlations, lags 1–4

For each lag in x and y, same-bin and cross-bin (Total at pixel *i* vs High at pixel *i+lag* — the classic charge-sharing signature, one photon split between neighbours and sorted into different bins):

```python
def lag_corr(A, B, dy, dx):
    a = A[:, 2:-2, 2:-2]
    b = B[:, 2+dy:B.shape[1]-2+dy, 2+dx:B.shape[2]-2+dx]
    az, bz = a - a.mean(0), b - b.mean(0)
    return (az*bz).mean(0) / (a.std(0)*b.std(0))
```

Mask any pair touching a bad pixel. Report mean ρ per lag, per direction, per bin combination.

The decay across lags is the discriminator: charge sharing dies by lag 1–2, K-fluorescence reaches further.

Standard error per pixel-pair is ~1/√1440 ≈ 0.026; averaged over thousands of pairs it's negligible. **A null result here is a real null, not an underpowered one.**

### (d) The flux trend

Plot each of (a), (b), (c) against ROI mean count rate. Six points per curve.

Reading it:

- **Flat across flux** → charge sharing and K-fluorescence dominate. These are geometric, set by the charge-cloud radius versus pixel pitch, so they shouldn't care about rate.
- **Rising with flux** → pileup is contributing. Two photons inside one shaping time sum into a single mis-energised count, which couples the bins and grows with rate.
- **Falling with flux** → suspect saturation or deadtime before believing anything physical.

A clear, non-flat trend is a *better* result than a large flat one: it means Σ(φ) is a real function with something to fit, which is checkpoint 2. A flat trend is still fine — Σ is then a constant, and the flux extension drops out of the paper without damaging the rest.

## Step 9 — Analytic propagation (checkpoint 3)

No training, no network, a few hundred lines of linear algebra. Do this in week 1 too — it's what makes the contribution a noise *model* rather than a training trick.

**Read Roessl 2007 and Tang 2023 first.** They've done the per-pixel version; you're extending to the cross-pixel term and validating empirically. Reproducing Tang's −1 for the two-material case is your correctness check, and if you *don't* get −1 something in your pipeline is wrong.

FDK is a **linear** operator. So a known projection-domain covariance maps to image-domain covariance in closed form, and the 2×2 decomposition maps that to material-image covariance in closed form. Chain them and you can predict, from Σ plus geometry alone, how noisy the shell and pulp maps will be — before training anything.

1. Take the measured Σ.
2. Propagate through FDK. The full matrix is astronomically large; use the standard local approach — local impulse response / noise power spectrum — to get the covariance in a neighbourhood around a voxel of interest.
3. Propagate through the 2×2 material inverse.
4. **Predict** material-image noise. Then reconstruct a real walnut, decompose it conventionally, and **measure** the material-image noise in a uniform region.
5. Compare. Also compute what a diagonal-Σ assumption would have predicted.

Two numbers come out, and they're the ones a reviewer will remember:

- **Prediction error** — how close the model gets. Small means you have a validated end-to-end noise model for PCD material decomposition, which stands as a contribution independent of any network.
- **Diagonal-assumption error** — by how much the field's current practice underestimates material-image noise. This is a claim about everyone else's method being wrong, which is more interesting than a claim about yours being right.

If the prediction misses badly, that discrepancy is itself a finding and tells you what physics the model is missing.

## Step 10 — Confound check (run in parallel)

The XCounter THOR has on-chip charge-summing / anti-coincidence logic. If it was enabled during acquisition, the chip already merges or rejects near-simultaneous adjacent hits — suppressing spatial correlation at the source and pushing Fano below 1. A near-zero ρ would then mean "hardware already fixed it," not "physics absent."

1. ~~Open `AcqPara.mat` and look for a detector-mode field.~~ **Done — there is no such field.** Full contents are tabulated under Step 0. This route is exhausted.
2. **Email the dataset authors. This is now the only way to answer the question.** Ask two things: whether anti-coincidence / charge-summing was enabled during the calibration acquisitions, and whether the setting was fixed or configurable.

Contacts: Enze Zhou (first author, ran the acquisition); corresponding authors Tianwu Xie `tianwuxie@fudan.edu.cn` and Qian Liu `qliu@hainanu.edu.cn`.

**If the mode turns out to be configurable**, this stops being purely a risk and becomes a contribution: the paper can characterise *when* covariance-aware training matters — hardware suppression on versus off. Worth asking explicitly whether the setting was fixed or adjustable, not just what it was set to.

---

## Verdict rule

| Outcome | Criterion | Action |
|---|---|---|
| **Go** | Any \|ρ\| ≳ 0.1 in the lag structure, **or** Fano clearly off 1, **or** \|rho_LH\| clearly nonzero | Proceed. The measured Σ becomes Figure 2. |
| **Kill** | Everything < ~0.05 after drift correction **and** Step 6 was clean **and** Step 9 shows the diagonal assumption costs < ~5% | Fall back to Gap #2. |
| **Ambiguous** | 0.05–0.1 | Let Step 9 decide. |

**On the ambiguous case:** Step 9 is the tiebreaker, and it's now part of the standard protocol rather than a contingency. If the diagonal assumption underestimates material-image noise by more than ~20%, the effect matters even at modest ρ — the ill-conditioned inverse does the amplifying for you. A small ρ with a large propagated consequence is still a paper; a small ρ with no propagated consequence is not.

---

## Extensions this protocol enables

Ordered by value per unit effort. The first two are folded into the steps above; the rest are downstream.

**Flux-conditioned Σ(φ) — in Step 5/8.** Σ as a function of local count rate rather than one fixed matrix, with the loss weight adapting per-pixel. **After the August 2026 sweep this is the least-occupied checkpoint and should carry the most weight in the paper.** Converts "I plugged in a matrix" into a noise model that generalises to operating points you never measured, and recovers the Gap #2 flux-conditioning idea from a single scan. Cost: one afternoon, but only if the per-ROI count rates are recorded during the kill test — otherwise it needs a reprocess.

**Analytic propagation — Step 9.** Covered above, now partly bounded by Roessl 2007 and Tang 2023. Still worth doing for the cross-pixel term and the empirical validation. Cost: a few hundred lines, no training.

**Per-pixel uncertainty output.** Once Σ exists, the network can output calibrated uncertainty on the material maps alongside the maps themselves. Nearly free, absent from current DL decomposition work, and the addition most likely to make a reviewer find the method *useful* rather than merely correct.

**Learned residual covariance.** Use the measured Σ as an initialisation, then let a small parameterised correction train end-to-end. Gives a measured-vs-measured+learned ablation and a story about the measurement being incomplete in ways the data reveals. Risk: another moving part, and a null correction is a null result inside your own paper.

**Anti-coincidence as an experimental variable.** Depends on the authors' reply — see Step 10.

### Deliberately excluded

- **Fancier architecture** (transformer, diffusion). Dilutes the claim, costs weeks, and invites the exact "the gain is from the architecture" criticism the ablation ladder exists to prevent.
- **Projection-domain one-step decomposition.** The PMMA holder is a third basis material in the beam path; the dataset authors flag this as why they avoided it.
- **A second dataset.** Every candidate has a licensing or fixed-operating-point problem. This was already settled when Gap #2 was demoted.

---

## Baselines (for later, recorded here so it doesn't get lost)

The ablation ladder carries the paper — same network, three losses: MSE, diagonal-weighted, full Σ. **The diagonal row is the critical one**; without it a reviewer says you only showed that noise weighting helps. If diagonal and full Σ tie, there is no contribution. Note the diagonal row is not a strawman — it reproduces the independent-Poisson objective of MICCAI 2024.

External comparators, priority order:

1. The dataset authors' own pipeline (FDK + Hann + 3D-TV + pixel-wise inversion). Table stakes.
2. Plain U-Net with MSE. Overlaps with ablation row 1 if you share the backbone, which is preferable.
3. PWLS / model-based decomposition — Tivnan, Wang & Stayman (arXiv 2010.01371); Long & Fessler, IEEE TMI 33(8):1614–1626 (2014); Schirra et al., IEEE TMI 32(7):1249–1257 (2013). Philosophically your direct ancestor: its statistical weight *is* a diagonal Σ.
4. One modern DL comparator — DIRECT-Net (Su et al., Med Phys 49(2):917–934, 2022) or a GAN-based decomposition (Ren et al., Comput Biol Med 180:108854, 2024; Guo et al., Nucl Sci Tech 34:45, 2023).

Cite but don't run: reconstruction networks (ULTRA, SOUL-Net, HITI-Net) and diffusion decomposition (Bousse's TDPS/ODPS, arXiv 2403.10183; Stayman's SDPS/JSDPS) — different problem scope, and reimplementation would eat the remaining weeks.

Useful quotes for the introduction, both from authors who saw the gap and left it:

- MD-Unet (Han et al., PLOS ONE 19:e0306627, 2024) note that because each bin's data represents attenuated photons at the same location, a spatial correlation exists — then train with MSE-based losses.
- Nadkarni et al. (PMC9969357) attribute residual distortion in their decomposition to the MSE loss and suggest future work test whether other loss functions remove it.

Also relevant: Persson's group (Eguizabal, Öktem, Persson, arXiv 2208.03360) is the adjacent group and a likely reviewer.

### Ground truth

There is no gold-standard shell/pulp map, and the dataset authors suggest users hand-annotate one — which is circular for evaluation. Two better options:

- **HAP phantom (primary quantitative).** QRM rods at known 50/100/200 mg/cm³ in PMMA. Decompose water/HAP, compare against true densities. Real quantitative validation, and HAP is a bone surrogate — reads medical to an IPMI reviewer in a way shell-vs-pulp does not. **Conditional on the Step 1 check that those projections are per-view, not averaged.**
- **Noise vs. dose on the walnuts.** Full-dose 1440-view decomposition as reference, sparse input as test. Measures exactly what the method claims to improve.

Shell/pulp then becomes the qualitative demonstration rather than the quantitative one.

---

## Implementation

Scripts live alongside this document in `killtest/`.

| File | Purpose |
|---|---|
| `kill_test.py` | Steps 1–8. Loads ROIs, masks bad pixels, drift-normalises, harmonic-detrends, computes all statistics, prints the verdict. CPU only, ~2 GB RAM, 15–25 min. |
| `plot_results.py` | Four figures. `fig1` is the paper's Figure 2; `fig2` is the decisive lag plot; `fig3` is checkpoint 2. |
| `make_synthetic.py` | Generates fake data with the real geometry and a *known* amount of injected correlation, for validating the analysis before the download lands. |

```bash
python kill_test.py --data <dir with Water_Phantom/ and HAP_Phantom/> \
                    --cal <dir with badchannelIndexAll.data> \
                    --phantom Water_Phantom --out results_water
python plot_results.py --results results_water --out figs_water
```

Run `HAP_Phantom` as well — that second operating point is the transfer test the IPMI framing depends on.

Everything in Step 6 is implemented: air ROIs are skipped for detrending, object ROIs get a two-harmonic fit, and the fraction of variance removed is printed per ROI so the wobble is visible rather than silently absorbed.

### Validated on synthetic data

400 views, real geometry (±27.5 channel wobble, drift, fixed-pattern gain, stuck columns), 20% horizontal charge sharing injected:

| | injected | null |
|---|---|---|
| peak correlation, air | **0.105** | 0.032 |
| peak correlation, object | **0.130** | 0.009 |
| nearest-neighbour TT | +0.105 | +0.001 |
| Fano (Total) | 0.948 | 0.999 |
| verdict | **GO** | **KILL** |

The neighbourhood map reproduces the injected *structure*, not just its magnitude — sharing was horizontal-only, and the map shows two hot pixels either side of centre with nothing above or below. `edge_L` had 95.5% of its variance removed as wobble and still returned a clean correlation afterwards.

Caveat: the synthetic's sharing model yields near-zero `rho_LH` by construction, so the cross-bin term is only weakly exercised. The spatial machinery is thoroughly tested; the cross-bin path is not.

### Two implementation notes that matter for interpretation

**`TH_x0` is excluded from every verdict.** High is a subset of Total, so their same-pixel correlation is trivially ~0.6–0.7 and carries no physics. Including it made every dataset an automatic GO in the first draft. `rho_LH` is the cross-bin number that means something.

**x-vs-y asymmetry is the detrending check.** The wobble is purely horizontal, so residual geometry leaks into x only, while real charge sharing is near-isotropic. `fig2` splits the directions specifically so this can be read off. A large asymmetry in the detrended curves means raising `N_HARMONICS` and re-running.

## Open decisions

- 14-week timeline for this direction (the existing table was built for Gap #2 and doesn't transfer). The framing above changes the allocation: checkpoints 1–3 are front-loaded and cheap, training is back-loaded, and checkpoint 2 now deserves more weight than originally planned.
- Whether the HAP phantom becomes the primary quantitative result (pending the Step 1 check).
- Which single modern DL comparator, and whether PWLS is worth the implementation cost.
- Whether to switch any of the five configuration choices above.

## Report back

**Reading, in parallel with the download:**

- **Tang 2023 §Method** (PMC9860003) — diagonal or general input covariance? Sets the checkpoint-3 boundary.
- **Winfree 2024** — their measured correlation values are the prior expectation for what this test should return on different hardware.

**Email today** — anti-coincidence mode, to Enze Zhou / Tianwu Xie / Qian Liu. No other route exists.

**Already resolved without downloading anything:**

| Question | Answer |
|---|---|
| HAP per-view or averaged? | **Per-view.** 1440 × 2 bins. |
| Header on the raw files? | **None.** Exactly 2,083,630 bytes. |
| Air ROI saturating? | **No.** 22 sigma of headroom. |
| Phantom on-axis? | **No — 1.2 mm off.** Detrending mandatory. |
| AcqPara detector mode? | **Field absent.** |

**Still open, needs the data:**

1. **Air-population correlations** — the cleanest number in the dataset, immune to the wobble. Run this first.
2. **Detrended object correlations** — must agree with air, or the detrending is suspect.
3. **Residual variance after detrending** — should fall to roughly Poisson. If not, something else is moving.

## Change log

**Aug 2026 — scripts written and validated.** `kill_test.py`, `plot_results.py`, `make_synthetic.py` added. Validated on synthetic data in both the injected-correlation and null cases. Two bugs caught during validation: `TH_x0` was inflating every verdict, and the lag plot averaged over direction, hiding the x/y asymmetry that distinguishes residual geometry from real physics. Both fixed.

**Aug 2026 — archive inspection (no full download).** Central directory and selected files pulled by HTTP range request. Record DOI updated (15738313 → 17328375). HAP_Phantom confirmed per-view → promoted to leading candidate for primary quantitative result. Step 2 gate passed in advance; measured single-view values added. Saturation warning **retracted** — 22σ headroom, air ROI stays in the statistics; the 4095 pixels are five stuck-high columns, now listed. AcqPara extracted and tabulated; `nFramesNumPerView=1` confirms noise is intact; no detector-mode field, so the author email is the sole remaining route. **Step 6 rewritten: the phantom is confirmed ~1.2 mm off-axis, geometry exceeds noise by ~6×, and harmonic detrending is now mandatory rather than conditional.** Measured ROI profile added to Step 5, with a new caveat that the flux axis is confounded with beam hardening.

**Aug 2026 — literature sweep.** Added prior-art position section. Measurement claim downgraded from "first" to "first use" (Winfree 2024). Propagation claim bounded by Roessl 2007 and Tang 2023, exact boundary pending a full-text read. MICCAI 2024 reclassified from competitor to supporting evidence after finding its objective is independent-Poisson. Philips patent reassessed as closer than previously characterised. Flux conditioning promoted to the primary novelty. Baselines and ground-truth options recorded.
