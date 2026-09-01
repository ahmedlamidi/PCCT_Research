# Walnut PCCT — measurement phase

Report: `REPORT.html`  (published artifact)

## Layout

    project/
      REPORT.html                  the write-up
      figures/                     six headline figures
      results/
        01_kill_test/              spatial-spectral covariance, water phantom
        02_test0/                  slab-calibration residual analysis
        03_reconstruction/         walnut reconstruction figures
        04_gate_A/                 cross-bin dependency (A2 calibration + A3 real data)
        05_gate_B/                 bed-position overlap sufficiency
        06_overlap_gradient/       disagreement vs local gradient
      code/                        all scripts, grouped by workstream
      logs/                        raw run logs

## Verdicts

| test | verdict | key number |
|---|---|---|
| Kill test | KILL | peak corr 0.0100 (air 0.0078) |
| Test 0 | mostly artifact | 96-98% of residual is tube drift |
| Gate A | FAIL | rho_LH below 0.0016 systematic floor |
| Gate B | MARGINAL | 17.8% double-covered |
| Overlap x gradient | not established | collapses at matched cone angle |

## Data (not copied here — large)

    data/CalibrationPhantomData/   water phantom + PMMA/Al slabs
    data/CalibrationTable/         air tables, STEPC, bad channels, HU table
    data/Walnut_1/                 4 bed positions, Total + High
    data/Walnut_2/                 (downloading — cross-sample check)

Intermediate volumes stay in `walnut/`, `gateB/`, `exp/` (~350 MB); they are
regenerable from `code/`.

## Reproducing

    python3 code/kill_test.py --data data/CalibrationPhantomData --cal data/CalibrationTable --out results_water
    python3 code/test0/analysis.py --full
    bash    code/gates/run_A2.sh
    python3 code/gates/run_A3.py
    python3 code/gates/run_B.py
    python3 code/walnut/run_all.py 1 2 3 4
    python3 code/overlap/analyse_matched.py 1
    python3 code/overlap/cone_matched.py 1

CPU only. `code/walnut/recon_tigre.py` is the CUDA path — written and geometry-verified,
but untested (no GPU on this machine).
