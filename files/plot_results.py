#!/usr/bin/env python3
"""
Visualise kill_test.py output.

    python plot_results.py --results results/ --out figs/

Produces:
  fig1_neighbourhood_maps.png   the 5x5 correlation structure, per ROI
  fig2_lag_decay.png            correlation vs lag  <- the decisive plot
  fig3_flux_trends.png          how everything moves with count rate
  fig4_summary.png              one-panel verdict figure
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

GO, KILL = 0.10, 0.05
plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25})


def load(d):
    with open(os.path.join(d, "results.json")) as f:
        r = json.load(f)
    meta = r.pop("_meta")
    names = list(r.keys())
    names.sort(key=lambda n: -r[n]["rate"])          # high flux -> low flux
    return r, meta, names


def fig_maps(res, names, meta, out):
    """5x5 neighbourhood correlation, three bin combinations per ROI."""
    L = meta["max_lag"]
    combos = [("TT", "Total-Total"), ("HH", "High-High"), ("TH", "Total-High")]
    fig, ax = plt.subplots(len(combos), len(names),
                           figsize=(1.55 * len(names), 5.0), squeeze=False)
    vmax = 0.0
    for n in names:
        for c, _ in combos:
            m = np.array(res[n]["maps"][c], float)
            m[L, L] = np.nan                          # blank the trivial centre
            vmax = max(vmax, np.nanmax(np.abs(m)))
    vmax = max(vmax, 0.02)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for i, (c, clabel) in enumerate(combos):
        for j, n in enumerate(names):
            m = np.array(res[n]["maps"][c], float)
            m[L, L] = np.nan
            a = ax[i][j]
            im = a.imshow(m, cmap="RdBu_r", norm=norm,
                          extent=[-L - .5, L + .5, L + .5, -L - .5])
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            if i == 0:
                a.set_title(f"{n}\n{res[n]['rate']:.0f} cts", fontsize=7)
            if j == 0:
                a.set_ylabel(clabel, fontsize=8)
            if not res[n]["detrended"]:
                for sp in a.spines.values():
                    sp.set_edgecolor("#2a9d8f"); sp.set_linewidth(2.0)

    fig.suptitle("Neighbourhood noise correlation  (centre pixel blanked; "
                 "green border = air ROI, no detrending applied)", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("correlation coefficient", fontsize=7)
    fig.savefig(os.path.join(out, "fig1_neighbourhood_maps.png"),
                bbox_inches="tight")
    plt.close(fig)


def fig_lags(res, names, meta, out):
    """Correlation vs lag. The decisive plot.

    Rows separate the two directions on purpose. The off-axis phantom wobble is
    purely HORIZONTAL, so residual geometry shows up in x only. Real charge
    sharing is near-isotropic. Comparing the two rows is therefore a direct
    check that the harmonic detrending worked: a large x/y asymmetry in the
    detrended (dashed) ROIs means geometry is still leaking through.
    """
    L = meta["max_lag"]
    se = meta["se_per_pair"]
    lags = np.arange(1, L + 1)
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.6), sharey=True, sharex=True)
    cmap = plt.cm.viridis(np.linspace(0, .88, len(names)))

    for row, (ax_lab, key) in enumerate([("x  (horizontal)", "x"),
                                         ("y  (vertical)", "y")]):
        for k, (c, title) in enumerate([("TT", "Total-Total"),
                                        ("HH", "High-High"),
                                        ("TH", "Total-High (cross-bin)")]):
            a = ax[row][k]
            for n, col in zip(names, cmap):
                v = [res[n]["lags"].get(f"{c}_{key}{d}") for d in lags]
                ls = "-" if not res[n]["detrended"] else "--"
                a.plot(lags, v, ls, marker="o", ms=3.5, color=col,
                       lw=1.4, label=f"{n} ({res[n]['rate']:.0f})")
            a.axhspan(-KILL, KILL, color="crimson", alpha=.07)
            a.axhline(GO, color="green", lw=.8, ls=":")
            a.axhline(-GO, color="green", lw=.8, ls=":")
            a.axhline(0, color="k", lw=.6)
            a.fill_between([lags[0], lags[-1]], -2 * se, 2 * se,
                           color="grey", alpha=.18, zorder=0)
            if row == 0:
                a.set_title(title, fontsize=9)
            if row == 1:
                a.set_xlabel("pixel lag")
            a.set_xticks(lags)
        ax[row][0].set_ylabel(f"correlation, {ax_lab}")
    ax[0][0].legend(fontsize=5.6, ncol=1, loc="best", framealpha=.9)
    fig.suptitle("Correlation decay with distance   —   solid = air (no "
                 "detrend), dashed = detrended;  grey band = 2 s.e.;  red = "
                 "kill zone\n"
                 "The wobble is horizontal only: a large x-vs-y asymmetry in "
                 "the dashed curves means residual geometry, not physics.",
                 fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_lag_decay.png"), bbox_inches="tight")
    plt.close(fig)


def fig_flux(res, names, out):
    """Flat with flux = geometric (charge sharing / K-fluorescence).
    Rising with flux = pileup. Falling = suspect saturation or deadtime."""
    rate = np.array([res[n]["rate"] for n in names])
    order = np.argsort(rate)
    r = rate[order]
    air = np.array([not res[n]["detrended"] for n in names])[order]

    def g(f):
        return np.array([f(res[n]) for n in names])[order]

    panels = [
        ("Fano factor (Total)", g(lambda d: d["fano_T"]), 1.0, "Poisson"),
        ("Fano factor (High)", g(lambda d: d["fano_H"]), 1.0, "Poisson"),
        (r"$\rho_{LH}$  (nesting removed)", g(lambda d: d["rho_LH"]), 0.0, "zero"),
        ("nearest-neighbour TT", g(lambda d: d["lags"]["TT_x1"]), 0.0, "zero"),
        ("nearest-neighbour HH", g(lambda d: d["lags"]["HH_x1"]), 0.0, "zero"),
        ("cross-bin neighbour TH", g(lambda d: d["lags"]["TH_x1"]), 0.0, "zero"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    for a, (t, v, ref, rl) in zip(ax.ravel(), panels):
        a.plot(r, v, "-", color="0.6", lw=1, zorder=1)
        a.scatter(r[~air], v[~air], s=34, c="#264653", label="object (detrended)",
                  zorder=3)
        a.scatter(r[air], v[air], s=52, c="#2a9d8f", marker="s",
                  edgecolor="k", lw=.5, label="air (no detrend)", zorder=4)
        a.axhline(ref, color="crimson", ls="--", lw=.9, label=rl)
        a.set_title(t, fontsize=8.5)
        a.set_xlabel("mean counts per view")
        a.set_xscale("log")
    ax[0][0].legend(fontsize=6.5, loc="best")
    fig.suptitle("Flux dependence   —   flat = geometric (charge sharing, "
                 "K-fluorescence);  rising = pileup;  falling = suspect "
                 "saturation/deadtime", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_flux_trends.png"), bbox_inches="tight")
    plt.close(fig)


def fig_summary(res, names, meta, out):
    se = meta["se_per_pair"]
    peaks, labels, isair = [], [], []
    for n in names:
        v = [abs(x) for k, x in res[n]["lags"].items()
             if x is not None and k != "TH_x0"]
        peaks.append(max(v) if v else 0.0)
        labels.append(f"{n}\n{res[n]['rate']:.0f} cts")
        isair.append(not res[n]["detrended"])
    peaks = np.array(peaks); isair = np.array(isair)

    fig, ax = plt.subplots(figsize=(1.05 * len(names) + 2, 3.6))
    cols = ["#2a9d8f" if a else "#264653" for a in isair]
    ax.bar(range(len(names)), peaks, color=cols)
    ax.axhline(GO, color="green", ls="--", lw=1.2)
    ax.axhline(KILL, color="crimson", ls="--", lw=1.2)
    ax.axhspan(0, KILL, color="crimson", alpha=.07)
    ax.axhspan(KILL, GO, color="orange", alpha=.07)
    ax.axhspan(GO, max(peaks.max() * 1.25, GO * 1.5), color="green", alpha=.07)
    ax.fill_between([-.6, len(names) - .4], 0, 2 * se, color="grey", alpha=.3)
    ax.text(len(names) - .45, GO, "  GO", color="green", va="bottom", fontsize=8)
    ax.text(len(names) - .45, KILL, "  KILL", color="crimson", va="top", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("peak |correlation| over all lags")
    ax.set_xlim(-.6, len(names) - .4)
    ax.set_title("Verdict per ROI   (teal = air, immune to the off-axis "
                 "wobble;  grey = 2 s.e.)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_summary.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res, meta, names = load(a.results)
    fig_maps(res, names, meta, a.out)
    fig_lags(res, names, meta, a.out)
    fig_flux(res, names, a.out)
    fig_summary(res, names, meta, a.out)
    print("wrote 4 figures to", a.out)
