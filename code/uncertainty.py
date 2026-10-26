"""
Foutmarges en correlaties voor de smeltberekening.

1. Gevoeligheid (one-at-a-time)
   De zomersmelt (JJA) wordt opnieuw berekend met verstoorde invoer en
   modelparameters. Meetnauwkeurigheden volgens Kuipers Munneke et al. (2018),
   tabel 2. De totale foutmarge is de wortel van de kwadratensom van de
   afzonderlijke effecten (aanname: onafhankelijke fouten).

2. Validatie met betrouwbaarheidsintervallen
   Berekende tegen gemeten smelt per maand: r (95%-BI via Fisher-z),
   gemiddelde afwijking (95%-BI via t-verdeling) en RMSE.

3. Correlaties zoals tabel 3 van het paper
   r tussen JJA-smelt en SWnet, LWnet, Rnet, SHF, LHF en albedo per station,
   met 95%-BI en p-waarde, voor alle jaren en alleen AWS type 0.
   Daarnaast de 95%-BI van de hoogtegradiënten (t-verdeling, n-2 vrijheidsgraden)
   en een bootstrap-BI voor het deel van de smeltvariantie dat door albedo komt.

Gebruik
-------
    python code/uncertainty.py [--out output/uncertainty]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import melt_comparison as mc
import elevation_albedo as ea

JJA = [6, 7, 8]
JJA_HOURS = 92 * 24
N_BOOT = 2000

# Verstoringen: naam -> (omschrijving, functie(df) of parameterwijziging)
SENSITIVITY = {
    "T ±0.4 °C": "meetnauwkeurigheid HMP45C (paper tabel 2)",
    "wind ±0.3 m/s": "meetnauwkeurigheid Young 05103 (paper tabel 2)",
    "SW ±5%": "kalibratie kortgolvig (CNR1, ~10% van dagtotaal netto straling)",
    "LWD ±5 W/m²": "kalibratie langgolvig (aanname)",
    "z0m ×2 / ÷2": "onzekerheid ruwheid (factor 2)",
    "meethoogte 2.5 / 4 m": "onbekende sensorhoogte type 0 (standaard 3 m)",
    "smeltdrempel 0.25 / 1 K": "drempel oppervlaktetemperatuur (standaard 0.5 K)",
    "G +10 / −10 W/m²": "warmtestroom in het ijs, standaard 0 (aanname; orde van grootte uit paper fig. 4)",
}

# Tabel 3 van Kuipers Munneke et al. (2018): r met JJA-smelt, en R² albedo (fig. 7)
PAPER_R = pd.DataFrame({
    "AWS5": {"SWnet": 0.57, "LWnet": 0.58, "Rnet": 0.86, "SHF": 0.78, "LHF": 0.88, "R2_albedo": 0.58},
    "AWS6": {"SWnet": 0.78, "LWnet": 0.66, "Rnet": 0.98, "SHF": 0.74, "LHF": 0.78, "R2_albedo": 0.87},
    "AWS9": {"SWnet": 0.88, "LWnet": 0.59, "Rnet": 0.98, "SHF": 0.71, "LHF": 0.60, "R2_albedo": 0.92},
    "AWS10": {"SWnet": 0.79, "LWnet": 0.70, "Rnet": 0.93, "SHF": 0.29, "LHF": 0.46, "R2_albedo": 0.76},
})


# ---------------------------------------------------------------------------
# Statistiek
# ---------------------------------------------------------------------------

def r_ci(x, y, alpha=0.05):
    """Pearson r met 95%-BI (Fisher-z) en p-waarde."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 4:
        return dict(r=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=n)
    r, p = stats.pearsonr(x[ok], y[ok])
    zc = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n - 3)
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    return dict(r=r, lo=np.tanh(z - zc), hi=np.tanh(z + zc), p=p, n=n)


def slope_ci(x, y, alpha=0.05):
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    s = stats.linregress(x[ok], y[ok])
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return dict(slope=s.slope, lo=s.slope - t * s.stderr, hi=s.slope + t * s.stderr, p=s.pvalue, n=n)


# ---------------------------------------------------------------------------
# 1. Gevoeligheid
# ---------------------------------------------------------------------------

def jja_melt(ts):
    """JJA-smelt (mm w.e.) en dekking per jaar."""
    j = ts[ts.index.month.isin(JJA)]
    dt_h = j.index.to_series().diff().shift(-1).dt.total_seconds().div(3600).clip(upper=1.0)
    dt_h = dt_h.fillna(dt_h.median())
    rate = j["M"] * 3600 / mc.LF
    g = rate.groupby(j.index.year)
    cov = (dt_h * j["M"].notna()).groupby(j.index.year).sum() / JJA_HOURS
    return g.mean() * JJA_HOURS, cov


def run(station, df, z0_factor=1.0, z_default=None, t_tol=None, g_flux=None):
    orig = (mc.z0m_series, mc.Z_DEFAULT, mc.T_MELT_TOL, mc.GROUND_FLUX)
    try:
        if z0_factor != 1.0:
            mc.z0m_series = lambda st, idx: orig[0](st, idx) * z0_factor
        if z_default is not None:
            mc.Z_DEFAULT = z_default
        if t_tol is not None:
            mc.T_MELT_TOL = t_tol
        if g_flux is not None:
            mc.GROUND_FLUX = g_flux
        return jja_melt(mc.melt_aws(station, df))[0]
    finally:
        mc.z0m_series, mc.Z_DEFAULT, mc.T_MELT_TOL, mc.GROUND_FLUX = orig


def sensitivity(station):
    df = mc.load_aws(station)
    base, cov = jja_melt(mc.melt_aws(station, df))
    years = cov.index[cov >= ea.MIN_JJA_COVERAGE]
    base = base.reindex(years)

    def mod(**cols):
        d = df.copy()
        for c, f in cols.items():
            d[c] = f(d[c])
        return d

    runs = {
        "T ±0.4 °C": (run(station, mod(TTT=lambda x: x + 0.4)), run(station, mod(TTT=lambda x: x - 0.4))),
        "wind ±0.3 m/s": (run(station, mod(ff=lambda x: x + 0.3)), run(station, mod(ff=lambda x: (x - 0.3).clip(lower=0)))),
        "SW ±5%": (run(station, mod(SWD=lambda x: x * 1.05, SWU=lambda x: x * 1.05)),
                   run(station, mod(SWD=lambda x: x * 0.95, SWU=lambda x: x * 0.95))),
        "LWD ±5 W/m²": (run(station, mod(LWD=lambda x: x + 5)), run(station, mod(LWD=lambda x: x - 5))),
        "z0m ×2 / ÷2": (run(station, df, z0_factor=2.0), run(station, df, z0_factor=0.5)),
        "meethoogte 2.5 / 4 m": (run(station, df, z_default=2.5), run(station, df, z_default=4.0)),
        "smeltdrempel 0.25 / 1 K": (run(station, df, t_tol=0.25), run(station, df, t_tol=1.0)),
        "G +10 / −10 W/m²": (run(station, df, g_flux=10.0), run(station, df, g_flux=-10.0)),
    }
    rows = []
    for name, (up, down) in runs.items():
        up, down = up.reindex(years), down.reindex(years)
        d_up, d_down = (up - base).mean(), (down - base).mean()
        rows.append({"station": station, "parameter": name, "toelichting": SENSITIVITY[name],
                     "delta_hoog_mm": d_up, "delta_laag_mm": d_down,
                     "halve_bandbreedte_mm": (abs(d_up) + abs(d_down)) / 2})
    out = pd.DataFrame(rows)
    out["halve_bandbreedte_pct"] = 100 * out["halve_bandbreedte_mm"] / base.mean()
    total = np.sqrt((out["halve_bandbreedte_mm"] ** 2).sum())
    summary = {"station": station, "n_jaren": len(years), "jja_smelt_mm": base.mean(),
               "foutmarge_mm": total, "foutmarge_pct": 100 * total / base.mean(),
               "grootste_bron": out.loc[out["halve_bandbreedte_mm"].idxmax(), "parameter"]}
    return out, summary


# ---------------------------------------------------------------------------
# 2. Validatie
# ---------------------------------------------------------------------------

def validation():
    d = pd.read_csv(mc.ROOT / "output" / "melt_per_station_M.csv")
    rows = []
    for st, m in d.groupby("station", sort=False):
        m = m.dropna(subset=["melt_mmwe", "obs_melt_mmwe_ice"])
        m = m[(m["melt_mmwe"] > 50) | (m["obs_melt_mmwe_ice"] > 50)]
        if len(m) < 5:
            continue
        diff = m["melt_mmwe"] - m["obs_melt_mmwe_ice"]
        rc = r_ci(m["melt_mmwe"].values, m["obs_melt_mmwe_ice"].values)
        t = stats.t.ppf(0.975, len(diff) - 1) * diff.std() / np.sqrt(len(diff))
        rows.append({"station": st, "n_maanden": len(m), "r": rc["r"], "r_lo": rc["lo"], "r_hi": rc["hi"],
                     "bias_mm": diff.mean(), "bias_lo": diff.mean() - t, "bias_hi": diff.mean() + t,
                     "rmse_mm": np.sqrt((diff**2).mean()),
                     "bias_pct_van_gemeten": 100 * diff.mean() / m["obs_melt_mmwe_ice"].mean()})
    return pd.DataFrame(rows).set_index("station")


# ---------------------------------------------------------------------------
# 3. Correlaties en gradiënten
# ---------------------------------------------------------------------------

def correlations(years):
    y = years.copy()
    y["M"] = y["melt"] * mc.LF / (JJA_HOURS * 3600)       # W/m²
    y["Rnet"] = y["SWnet"] + y["LWnet"]
    rows = []
    for subset, sel in [("alle jaren", y), ("alleen type 0", y[y["aws_type"] == 0])]:
        for st in mc.AWS_STATIONS:
            s = sel[sel["station"] == st]
            for v in ["SWnet", "LWnet", "Rnet", "SHF", "LHF", "albedo"]:
                rc = r_ci(s[v].values, s["M"].values)
                rows.append({"subset": subset, "station": st, "variabele": v, **rc,
                             "paper_r": (-np.sqrt(PAPER_R.loc["R2_albedo", st]) if v == "albedo"
                                         else PAPER_R.loc[v, st])})
    return pd.DataFrame(rows)


def gradient_ci(years):
    clim = years.groupby("station").mean(numeric_only=True)
    rows = []
    for v in ["T", "albedo", "SWnet", "SHF", "melt"]:
        s = slope_ci(clim["elevation"].values, clim[v].values)
        rows.append({"variabele": v, "gradient_per_100m": s["slope"] * 100,
                     "lo": s["lo"] * 100, "hi": s["hi"] * 100, "p": s["p"], "n": s["n"]})
    return pd.DataFrame(rows).set_index("variabele")


def albedo_share_ci(years, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for st in mc.AWS_STATIONS:
        d = years[years["station"] == st].dropna(subset=["melt", "melt_clim_alb"])
        m, mc_ = d["melt"].values, d["melt_clim_alb"].values
        share = 1 - mc_.var(ddof=1) / m.var(ddof=1)
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(d), len(d))
            if m[i].var() > 0:
                boot.append(1 - mc_[i].var(ddof=1) / m[i].var(ddof=1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"station": st, "n_jaren": len(d), "aandeel_albedo": share, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).set_index("station")


# ---------------------------------------------------------------------------
# Figuur
# ---------------------------------------------------------------------------

def plot_sensitivity(sens, summary, out):
    params = list(SENSITIVITY)
    fig, axes = plt.subplots(1, len(mc.AWS_STATIONS), figsize=(13, 4.7), sharey=True)
    for ax, st in zip(axes, mc.AWS_STATIONS):
        s = sens[sens["station"] == st].set_index("parameter").reindex(params)
        base = summary.loc[st, "jja_smelt_mm"]
        y = np.arange(len(params))
        ax.barh(y + 0.2, 100 * s["delta_hoog_mm"] / base, height=0.38, color=mc.COLORS[st], label="first value")
        ax.barh(y - 0.2, 100 * s["delta_laag_mm"] / base, height=0.38, color="#a8a7a2", label="second value")
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.set_yticks(y, ["T +0.4 / −0.4 °C", "wind +0.3 / −0.3 m/s", "SW +5 / −5%", "LWD +5 / −5 W/m²",
                          "z0m ×2 / ÷2", "sensor height 2.5 / 4 m", "melt threshold 0.25 / 1 K",
                          "ground heat flux +10 / −10 W/m²"])
        ax.invert_yaxis()
        ax.set_title(f"{st}  (±{summary.loc[st, 'foutmarge_pct']:.0f}% total)", loc="left", fontsize=10)
        ax.set_xlabel("change in JJA melt (%)")
        mc.style(ax)
        ax.grid(axis="x", color="#e5e4e0", lw=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc="lower center")
    fig.suptitle("Sensitivity of summer melt to input errors and model choices", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(mc.ROOT / "output" / "uncertainty"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    parts, summ = [], []
    for st in mc.AWS_STATIONS:
        print(f"{st}: gevoeligheidsruns ...", flush=True)
        s, sm = sensitivity(st)
        parts.append(s)
        summ.append(sm)
    sens = pd.concat(parts)
    summary = pd.DataFrame(summ).set_index("station")
    sens.round(1).to_csv(out / "sensitivity.csv", index=False)
    summary.round(1).to_csv(out / "uncertainty_summary.csv")
    # foutmarge zonder de modelkeuze voor de smeltdrempel (alleen meetfouten en ruwheid)
    meas = sens[~sens["parameter"].str.startswith("smeltdrempel")]
    summary["foutmarge_zonder_drempel_pct"] = (meas.groupby("station")["halve_bandbreedte_pct"]
                                               .apply(lambda x: np.sqrt((x**2).sum())))
    summary.round(1).to_csv(out / "uncertainty_summary.csv")
    print("\nFoutmarge JJA-smelt:"); print(summary.round(1).to_string())
    print(sens.pivot(index="parameter", columns="station", values="halve_bandbreedte_pct")
          .reindex(columns=mc.AWS_STATIONS).round(1).to_string())

    val = validation()
    val.round(2).to_csv(out / "validation_ci.csv")
    print("\nValidatie (maand):"); print(val.round(2).to_string())

    years = pd.read_csv(mc.ROOT / "output" / "elevation" / "jja_per_station_year.csv", index_col=0)
    cor = correlations(years)
    cor.round(3).to_csv(out / "correlations.csv", index=False)
    print("\nCorrelaties met JJA-smelt (r [95%-BI]):")
    cor["txt"] = cor.apply(lambda r: f"{r.r:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]", axis=1)
    print(cor.pivot_table(index=["subset", "variabele"], columns="station", values="txt", aggfunc="first")
          .reindex(columns=mc.AWS_STATIONS).to_string())

    gr = gradient_ci(years)
    gr.round(3).to_csv(out / "gradient_ci.csv")
    print("\nHoogtegradiënten met 95%-BI:"); print(gr.round(3).to_string())

    ab = albedo_share_ci(years)
    ab.round(2).to_csv(out / "albedo_share_ci.csv")
    print("\nAandeel albedo in smeltvariantie (bootstrap 95%-BI):"); print(ab.round(2).to_string())

    plot_sensitivity(sens, summary, out / "sensitivity.png")
    print(f"\nOutput opgeslagen in {out}")


if __name__ == "__main__":
    main()
