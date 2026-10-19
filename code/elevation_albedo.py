"""
Onderzoekslijn B: hoogte-afhankelijkheid van de energiebalans en de albedo-feedback.

1. Hoogtegradiënten
   Per station en per zomer (JJA) de gemiddelde meteorologie (T, wind, RH),
   straling, albedo, SEB-componenten en smelt. Daarna lineaire regressie
   tegen de hoogte, zowel over de hele periode per station als over de
   gemeenschappelijke periode waarin alle vier de AWS'en data hebben.
   Per jaar wordt ook de smeltgradiënt (mm w.e. per 100 m) bepaald.

2. Langetermijntrends
   Trend per station (2003-2021) van JJA-temperatuur, albedo en smelt.

3. Albedo-experiment
   De smelt wordt opnieuw berekend met een klimatologisch albedo: SWU wordt
   vervangen door SWD * albedo_clim(dag-van-het-jaar), met albedo_clim de
   mediaan over alle jaren (15-daags gladgestreken). Het verschil
       dM_albedo = M(werkelijk albedo) - M(klimatologisch albedo)
   is de directe bijdrage van de albedo-anomalie aan de smelt in dat jaar.
   Let op: oppervlaktetemperatuur en turbulente fluxen blijven gelijk, dus
   terugkoppelingen via die termen zitten er niet in.

Gebruik
-------
    python code/elevation_albedo.py [--out output/elevation]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import melt_comparison as mc

JJA = [6, 7, 8]
MIN_JJA_COVERAGE = 0.8
ALB_SMOOTH_DAYS = 15

VARS = {  # kolom: (label, eenheid)
    "T": ("Air temperature", "°C"),
    "ff": ("Wind speed", "m/s"),
    "albedo": ("Albedo", "-"),
    "SWD": ("Incoming shortwave radiation", "W/m²"),
    "LWD": ("Incoming longwave radiation", "W/m²"),
    "SWnet": ("Net shortwave radiation", "W/m²"),
    "LWnet": ("Net longwave radiation", "W/m²"),
    "SHF": ("Sensible heat flux", "W/m²"),
    "LHF": ("Latent heat flux", "W/m²"),
    "melt": ("JJA melt", "mm w.e."),
}


def station_elevation(station):
    f = sorted((mc.DATA / station).glob("*.tab"))[0]
    txt = f.read_text(encoding="utf-8")[:20000]
    return float(re.search(r"ELEVATION: ([0-9.]+) m \* LOCATION", txt).group(1))


# ---------------------------------------------------------------------------
# Albedo
# ---------------------------------------------------------------------------

def daily_albedo(df):
    """Dagalbedo als verhouding van dagsommen (vermijdt zonnestandeffecten)."""
    swd = df["SWD"].clip(lower=0)
    swu = df["SWU"].clip(lower=0)
    ok = swd.notna() & swu.notna()
    d = pd.DataFrame({"swd": swd.where(ok), "swu": swu.where(ok)}).resample("D").sum(min_count=1)
    alb = (d["swu"] / d["swd"]).where(d["swd"] > 24 * 30)   # minimaal ~30 W/m² daggemiddeld
    return alb.where(alb.between(0.1, 0.95))


def climatological_albedo(alb_daily):
    clim = alb_daily.groupby(alb_daily.index.dayofyear).median()
    clim = clim.reindex(range(1, 367))
    return clim.rolling(ALB_SMOOTH_DAYS, center=True, min_periods=5).mean()


def melt_with_clim_albedo(station, df, clim):
    df2 = df.copy()
    a = clim.reindex(df2.index.dayofyear).values
    swu_clim = df2["SWD"].clip(lower=0) * a
    df2["SWU"] = np.where(np.isnan(a), df2["SWU"], swu_clim)
    return mc.melt_aws(station, df2)


# ---------------------------------------------------------------------------
# Zomerstatistieken
# ---------------------------------------------------------------------------

def jja_summary(station):
    df = mc.load_aws(station)
    seb = mc.melt_aws(station, df)
    alb_d = daily_albedo(df)
    clim = climatological_albedo(alb_d)
    seb_clim = melt_with_clim_albedo(station, df, clim)

    sel = df.index.month.isin(JJA)
    df, seb, seb_clim = df[sel], seb[sel], seb_clim[sel]
    alb_d = alb_d[alb_d.index.month.isin(JJA)]

    agg = mc.aggregate(seb, "Y", min_coverage=0)
    agg_clim = mc.aggregate(seb_clim, "Y", min_coverage=0)
    # aggregate() rekent dekking t.o.v. het hele jaar; omrekenen naar JJA (92 dagen)
    jja_hours = 92 * 24
    year_hours = pd.Series(agg.index.to_timestamp().is_leap_year, agg.index).map({True: 8784, False: 8760})
    cov = agg["coverage"] * year_hours / jja_hours
    good = cov >= MIN_JJA_COVERAGE

    # melt_mmwe is geschaald naar het hele jaar; terugschalen naar JJA-som
    scale = jja_hours / year_hours
    yrs = agg.index.year
    g = df.groupby(df.index.year)
    out = pd.DataFrame({
        "station": station,
        "T": g["TTT"].mean().reindex(yrs).values,
        "ff": g["ff"].mean().reindex(yrs).values,
        "RH": g["RH"].mean().reindex(yrs).values,
        "SWD": g["SWD"].mean().reindex(yrs).values,
        "LWD": g["LWD"].mean().reindex(yrs).values,
        "albedo": alb_d.groupby(alb_d.index.year).mean().reindex(yrs).values,
        "SWnet": agg["SWnet"].values, "LWnet": agg["LWnet"].values,
        "SHF": agg["SHF"].values, "LHF": agg["LHF"].values,
        "melt": (agg["melt_mmwe"] * scale).values,
        "melt_clim_alb": (agg_clim["melt_mmwe"] * scale).values,
        "coverage_jja": cov.clip(upper=1).round(2).values,
        # stationstype (0 = oud, 1 = compact IMAU type); wissel geeft sprongen in T en wind
        "aws_type": g["ID"].mean().round().reindex(yrs).values,
    }, index=pd.Index(yrs, name="year"))
    return out[good.values], clim


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def regress(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, p=np.nan, n=int(ok.sum()))
    s = stats.linregress(x[ok], y[ok])
    return dict(slope=s.slope, intercept=s.intercept, r2=s.rvalue**2, p=s.pvalue, n=int(ok.sum()))


def elevation_gradients(clim_tab):
    rows = []
    for v in VARS:
        r = regress(clim_tab["elevation"].values, clim_tab[v].values)
        rows.append({"variabele": v, "eenheid": VARS[v][1],
                     "gradient_per_100m": r["slope"] * 100, "r2": r["r2"], "p": r["p"], "n": r["n"]})
    return pd.DataFrame(rows).set_index("variabele")


def yearly_melt_gradient(years, elev):
    rows = []
    for y, d in years.groupby(level="year"):
        d = d.dropna(subset=["melt"])
        if len(d) < 3:
            continue
        r = regress(d["station"].map(elev).values.astype(float), d["melt"].values)
        rows.append({"year": y, "n_stations": len(d), "melt_gradient_per_100m": r["slope"] * 100,
                     "r2": r["r2"], "stations": ",".join(d["station"])})
    return pd.DataFrame(rows).set_index("year")


def trends(years):
    """Trends over alle jaren en (homogeniteitscheck) alleen over AWS type 0."""
    rows = []
    for st, d_all in years.groupby("station"):
        for subset, d in [("alle jaren", d_all), ("alleen type 0", d_all[d_all["aws_type"] == 0])]:
            if len(d) < 5:
                continue
            for v in ["T", "ff", "albedo", "melt", "SHF", "SWnet"]:
                r = regress(d.index.values.astype(float), d[v].values)
                rows.append({"station": st, "subset": subset, "variabele": v,
                             "trend_per_decade": r["slope"] * 10, "p": r["p"], "n": r["n"],
                             "periode": f"{d.index.min()}-{d.index.max()}"})
    return pd.DataFrame(rows)


def type_change_check(years, ref="AWS6"):
    """Sprong bij stationswissel: verschil tussen type 1 en type 0, ook relatief t.o.v. een referentie."""
    rows = []
    for st, d in years.groupby("station"):
        if d["aws_type"].nunique() < 2:
            continue
        for v in ["T", "ff", "SHF"]:
            jump = d.loc[d["aws_type"] == 1, v].mean() - d.loc[d["aws_type"] == 0, v].mean()
            row = {"station": st, "variabele": v, "sprong_type1_min_type0": jump}
            if st != ref:
                r = years[years["station"] == ref][v]
                diff = (d[v] - r).dropna()
                t = d["aws_type"].reindex(diff.index)
                row[f"sprong_relatief_tov_{ref}"] = diff[t == 1].mean() - diff[t == 0].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def albedo_attribution(years):
    rows = []
    for st, d in years.groupby("station"):
        d = d.dropna(subset=["melt", "melt_clim_alb"])
        dm_alb = d["melt"] - d["melt_clim_alb"]
        anom = d["melt"] - d["melt"].mean()
        rows.append({
            "station": st, "n_jaren": len(d),
            "gem_smelt": d["melt"].mean(),
            "std_smelt": d["melt"].std(),
            "std_smelt_clim_albedo": d["melt_clim_alb"].std(),
            # fractie van de variantie die verdwijnt als het albedo klimatologisch is
            "variantie_door_albedo": 1 - d["melt_clim_alb"].var() / d["melt"].var(),
            "r_anomalie_albedobijdrage": np.corrcoef(anom, dm_alb)[0, 1],
            "max_albedobijdrage": dm_alb.max(), "jaar_max": int(dm_alb.idxmax()),
            "min_albedobijdrage": dm_alb.min(), "jaar_min": int(dm_alb.idxmin()),
        })
    return pd.DataFrame(rows).set_index("station")


# ---------------------------------------------------------------------------
# Figuren
# ---------------------------------------------------------------------------

def style(ax):
    mc.style(ax)
    ax.grid(axis="x", color="#e5e4e0", lw=0.8)


def plot_gradients(clim_full, clim_common, grads, out):
    panels = ["T", "albedo", "SWnet", "SHF", "LWnet", "melt"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, v in zip(axes.flat, panels):
        for st, row in clim_full.iterrows():
            ax.scatter(row[v], row["elevation"], s=60, color=mc.COLORS[st], zorder=3,
                       edgecolor="white", lw=1.5, label=st)
            ax.annotate(st, (row[v], row["elevation"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color="#52514e")
        ax.scatter(clim_common[v], clim_common["elevation"], s=28, marker="D",
                   facecolor="none", edgecolor="#52514e", lw=1, zorder=4)
        g = grads.loc[v]
        z = np.linspace(clim_full["elevation"].min(), clim_full["elevation"].max(), 10)
        r = regress(clim_full["elevation"].values, clim_full[v].values)
        ax.plot(r["intercept"] + r["slope"] * z, z, color="#a8a7a2", lw=1.2, ls="--", zorder=2)
        unit = "" if VARS[v][1] == "-" else f" {VARS[v][1]}"
        ax.set_title(f"{VARS[v][0]}\n{g.gradient_per_100m:+.3g}{unit} per 100 m  (R²={g.r2:.2f})",
                     loc="left", fontsize=10)
        ax.set_xlabel(VARS[v][1])
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("elevation (m)")
    fig.text(0.01, 0.005, "● JJA mean over all years per station   ◇ common period "
             "(all 4 stations)   -- linear fit", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_partition(clim_full, out):
    comp = ["SWnet", "LWnet", "SHF", "LHF"]
    ccol = {"SWnet": "#2a78d6", "LWnet": "#eb6834", "SHF": "#1baf7a", "LHF": "#eda100"}
    d = clim_full.sort_values("elevation")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(comp)
    xs = np.arange(len(d))
    for i, c in enumerate(comp):
        ax.bar(xs + (i - 1.5) * w, d[c], width=w * 0.92, color=ccol[c], label=c)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(xs, [f"{s}\n{int(e)} m" for s, e in zip(d.index, d["elevation"])])
    ax.set_ylabel("W m$^{-2}$ (JJA mean)")
    ax.set_title("Summer surface energy balance components, from low to high elevation", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    mc.style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_albedo_experiment(years, out):
    sts = [s for s in mc.AWS_STATIONS if s in years["station"].unique()]
    fig, axes = plt.subplots(len(sts), 1, figsize=(11, 2.4 * len(sts)), sharex=True)
    for ax, st in zip(np.atleast_1d(axes), sts):
        d = years[years["station"] == st].dropna(subset=["melt", "melt_clim_alb"])
        mean = d["melt"].mean()
        ax.bar(d.index - 0.2, d["melt"] - mean, width=0.38, color=mc.COLORS[st],
               label="melt anomaly (observed albedo)")
        ax.bar(d.index + 0.2, d["melt"] - d["melt_clim_alb"], width=0.38, color="#a8a7a2",
               label="albedo anomaly contribution")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(f"{st}  (mean JJA melt {mean:.0f} mm w.e.)", loc="left", fontsize=10)
        ax.set_ylabel("mm w.e.")
        mc.style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_albedo_climatology(clims, years, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for st, c in clims.items():
        c = c.loc[120:270]
        ax1.plot(c.index, c.values, lw=2, color=mc.COLORS[st], label=st)
    ax1.set_xlabel("day of year")
    ax1.set_ylabel("albedo")
    ax1.set_title("Climatological albedo (median, 15-day smoothed)", loc="left")
    ax1.legend(frameon=False)
    mc.style(ax1)
    for st in mc.AWS_STATIONS:
        d = years[years["station"] == st]
        ax2.scatter(d["albedo"], d["melt"], s=30, color=mc.COLORS[st], edgecolor="white", lw=0.5, label=st)
        r = regress(d["albedo"].values, d["melt"].values)
        if np.isfinite(r["slope"]):
            x = np.linspace(d["albedo"].min(), d["albedo"].max(), 5)
            ax2.plot(x, r["intercept"] + r["slope"] * x, color=mc.COLORS[st], lw=1.2)
    ax2.set_xlabel("mean JJA albedo")
    ax2.set_ylabel("JJA melt (mm w.e.)")
    ax2.set_title("Summer melt vs summer albedo (per year)", loc="left")
    ax2.legend(frameon=False)
    style(ax2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(mc.ROOT / "output" / "elevation"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    elev = {st: station_elevation(st) for st in mc.AWS_STATIONS}
    parts, clims = [], {}
    for st in mc.AWS_STATIONS:
        print(f"{st} ({elev[st]:.0f} m): zomerstatistieken en albedo-experiment ...")
        y, clims[st] = jja_summary(st)
        parts.append(y)
    years = pd.concat(parts)
    years["elevation"] = years["station"].map(elev)
    years.round(3).to_csv(out / "jja_per_station_year.csv")

    num = [c for c in years.columns if c not in ("station",)]
    clim_full = years.groupby("station")[num].mean().reindex(mc.AWS_STATIONS)
    common_years = sorted(set.intersection(*[set(d.index) for _, d in years.groupby("station")]))
    clim_common = years.loc[years.index.isin(common_years)].groupby("station")[num].mean().reindex(mc.AWS_STATIONS)
    clim_full.round(3).to_csv(out / "jja_climatology_full.csv")
    clim_common.round(3).to_csv(out / "jja_climatology_common.csv")

    grads_full = elevation_gradients(clim_full)
    grads_common = elevation_gradients(clim_common)
    grads = grads_full.join(grads_common, rsuffix="_common")
    grads.round(4).to_csv(out / "elevation_gradients.csv")
    ygrad = yearly_melt_gradient(years, elev)
    ygrad.round(2).to_csv(out / "melt_gradient_per_year.csv")
    tr = trends(years)
    tr.round(4).to_csv(out / "trends.csv", index=False)
    attr = albedo_attribution(years)
    attr.round(3).to_csv(out / "albedo_attribution.csv")

    pd.set_option("display.width", 160)
    print(f"\nGemeenschappelijke jaren: {common_years}")
    print("\nHoogtegradiënten JJA (per 100 m):")
    print(grads[["eenheid", "gradient_per_100m", "r2", "gradient_per_100m_common", "r2_common"]].round(3).to_string())
    print("\nSmeltgradiënt per jaar (mm w.e. per 100 m):")
    print(ygrad.round(1).to_string())
    print("\nTrends per decennium:")
    print(tr.pivot_table(index=["station", "subset"], columns="variabele",
                         values="trend_per_decade").round(3).to_string())
    print("\nSignificant (p < 0.1):")
    print(tr[tr["p"] < 0.1][["station", "subset", "variabele", "trend_per_decade", "p", "periode"]]
          .round(3).to_string(index=False))
    tc = type_change_check(years)
    tc.round(2).to_csv(out / "aws_type_change_check.csv", index=False)
    print("\nSprong bij wissel stationstype (gemiddelde type 1 - type 0):")
    print(tc.round(2).to_string(index=False))
    print("\nAlbedo-experiment:")
    print(attr.round(2).to_string())

    plot_gradients(clim_full, clim_common, grads_full, out / "elevation_gradients.png")
    plot_partition(clim_full, out / "seb_partition_elevation.png")
    plot_albedo_experiment(years, out / "albedo_contribution.png")
    plot_albedo_climatology(clims, years, out / "albedo_climatology.png")
    print(f"\nOutput opgeslagen in {out}")


if __name__ == "__main__":
    main()
