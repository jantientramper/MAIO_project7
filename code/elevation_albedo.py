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


def albedo_spread(alb_daily, q=(0.25, 0.75)):
    """Spreiding tussen jaren per dag van het jaar (percentielen), net zo gladgestreken."""
    g = alb_daily.groupby(alb_daily.index.dayofyear)
    return tuple(g.quantile(x).reindex(range(1, 367))
                 .rolling(ALB_SMOOTH_DAYS, center=True, min_periods=5).mean() for x in q)


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
    yrs_alb = alb_d.dropna().index.year
    clim.attrs["period"] = f"{yrs_alb.min()}–{yrs_alb.max()}"   # jaren in de klimatologie
    clim.attrs["p25"], clim.attrs["p75"] = albedo_spread(alb_d)  # variatie tussen jaren
    seb_clim = melt_with_clim_albedo(station, df, clim)

    sel = df.index.month.isin(JJA)
    # df is per (half) uur, de gekoppelde smelt per uur: masker per tabel
    df = df[sel]
    seb = seb[seb.index.month.isin(JJA)]
    seb_clim = seb_clim[seb_clim.index.month.isin(JJA)]
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
        "G": agg["G"].values if "G" in agg else np.nan,
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


# Vorm per station en kleur per serie voor de hoogtefiguur
STATION_MARKERS = {"AWS5": "o", "AWS6": "s", "AWS9": "^", "AWS10": "*"}
MARKER_SIZE = {"*": 170}   # ster oogt klein, daarom groter
SERIES_COLORS = {"SWnet": "#2a78d6", "LWnet": "#eb6834", "SHF": "#1baf7a", "melt": "#6250d6"}
NEUTRAL = "#52514e"


def _gradient_series(ax, clim, v, color, label):
    """Punten per station (eigen vorm) plus lineaire fit, in een kleur."""
    for st, row in clim.iterrows():
        m = STATION_MARKERS[st]
        ax.scatter(row[v], row["elevation"], s=MARKER_SIZE.get(m, 80), marker=m, color=color,
                   edgecolor="white", lw=1, zorder=3)
    r = regress(clim["elevation"].values, clim[v].values)
    z = np.linspace(clim["elevation"].min(), clim["elevation"].max(), 10)
    line, = ax.plot(r["intercept"] + r["slope"] * z, z, color=color, lw=2, ls="--", zorder=2, label=label)
    return line


def _station_legend(clim):
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="", marker=STATION_MARKERS[st], ms=13 if STATION_MARKERS[st] == "*" else 9,
                   color=NEUTRAL, mec="white", label=f"{st} ({int(clim.loc[st, 'elevation'])} m)")
            for st in clim.index]


def plot_gradients(clim_full, clim_common, grads, out):
    """Twee figuren: temperatuur en albedo (out_T_albedo) en energiebalans + smelt (out)."""
    # --- figuur 1: temperatuur en albedo ---
    fig, (ax_t, ax_a) = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    for a, v in [(ax_t, "T"), (ax_a, "albedo")]:
        _gradient_series(a, clim_full, v, NEUTRAL, None)
        g = grads.loc[v]
        unit = "" if VARS[v][1] == "-" else f" {VARS[v][1]}"
        a.set_title(f"{VARS[v][0]}\n{g.gradient_per_100m:+.3g}{unit} per 100 m  (R²={g.r2:.2f})", fontsize=10)
        a.set_xlabel(VARS[v][0] if VARS[v][1] == "-" else f"{VARS[v][0]} ({VARS[v][1]})")
        style(a)
    ax_t.set_ylabel("Elevation (m)")
    fig.legend(handles=_station_legend(clim_full), frameon=False, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.06), ncol=4)
    fig.text(0.01, 0.015, "Points: JJA mean over all years per station.  Dashed: linear fit.",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out.with_name("elevation_T_albedo.png"), dpi=600)
    plt.close(fig)

    # --- figuur 2: energiebalanstermen (onderste x-as) en smelt (bovenste x-as) ---
    fig, ax = plt.subplots(figsize=(9, 7.2))

    def lab(v):
        g = grads.loc[v]
        return f"{VARS[v][0]}: {g.gradient_per_100m:+.3g} {VARS[v][1]} per 100 m (R²={g.r2:.2f})"

    lines = [_gradient_series(ax, clim_full, v, SERIES_COLORS[v], lab(v)) for v in ["SWnet", "LWnet", "SHF"]]
    ax.axvline(0, color="#a8a7a2", lw=0.8, zorder=1)
    ax.set_xlabel("Energy flux, JJA mean (W/m²)")
    ax.set_ylabel("Elevation (m)")
    style(ax)
    top = ax.twiny()
    melt_line = _gradient_series(top, clim_full, "melt", SERIES_COLORS["melt"], lab("melt"))
    lines.append(melt_line)
    # schaal van de bovenste as: de smeltfit valt precies midden tussen de fits van SHF en
    # SWnet (op de laagste en hoogste stationshoogte), zodat de lijnen elkaar niet kruisen
    z_ends = np.array([clim_full["elevation"].min(), clim_full["elevation"].max()])
    def fit_at(v):
        r = regress(clim_full["elevation"].values, clim_full[v].values)
        return r["intercept"] + r["slope"] * z_ends
    w_target = (fit_at("SHF") + fit_at("SWnet")) / 2          # W/m² op de onderste as
    m_fit = fit_at("melt")                                      # mm w.e. op de bovenste as
    b = (w_target[1] - w_target[0]) / (m_fit[1] - m_fit[0])     # W/m² per mm w.e.
    a = w_target[0] - b * m_fit[0]
    x0, x1 = ax.get_xlim()
    top.set_xlim((x0 - a) / b, (x1 - a) / b)
    top.set_xticks([t for t in top.get_xticks() if 0 <= t <= top.get_xlim()[1]])   # geen negatieve smelt
    # bovenste as in de kleur van de smeltlijn, zodat duidelijk is welke serie erbij hoort
    melt_color = melt_line.get_color()
    top.set_xlabel("JJA melt (mm w.e.)", color=melt_color)
    top.tick_params(axis="x", colors=melt_color)
    top.spines["top"].set_color(melt_color)
    ax.set_title("Surface energy balance and melt vs elevation", fontsize=11, pad=12)

    # legenda's: kleur = serie, vorm = station
    fig.legend(handles=_station_legend(clim_full), frameon=False, fontsize=9, loc="lower left",
               bbox_to_anchor=(0.01, 0.175), ncol=4, title="Station (marker)", title_fontsize=9)
    fig.legend(handles=lines, frameon=False, fontsize=9, loc="lower left",
               bbox_to_anchor=(0.01, 0.03), ncol=1, title="Series (colour)", title_fontsize=9)
    fig.text(0.99, 0.015, "Points: JJA mean over all years per station.  Dashed: linear fit.",
             fontsize=8, color="#52514e", ha="right")
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    fig.savefig(out, dpi=600)
    plt.close(fig)


def plot_partition(clim_full, out):
    comp = ["SWnet", "LWnet", "SHF", "LHF"]
    ccol = {
    "SWnet": "#12A1CD",
    "LWnet": "#EABE2E",
    "SHF":   "#50B57F",
    "LHF":   "#BB69DE"}
    # ccol = {"SWnet": "#278BB6", "LWnet": "#D7CF31", "SHF": "#33B422", "LHF": "#CE48A8"}
    d = clim_full.sort_values("elevation")
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(comp)
    xs = np.arange(len(d))
    for i, c in enumerate(comp):
        ax.bar(xs + (i - 1.5) * w, d[c], width=w * 0.92, color=ccol[c], label=c)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(xs, [rf"$\mathbf{{{s}}}$" + f"\n{int(e)} m"for s, e in zip(d.index, d["elevation"])], fontsize=11)
    ax.set_ylabel("W m$^{-2}$ [JJA mean]", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title("Summer SEB components",fontsize=13)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=11)
    mc.style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
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
        ax.set_title(f"{st}  (mean JJA melt {mean:.0f} mm w.e.)", fontsize=10)
        ax.set_ylabel("mm w.e.")
        mc.style(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)


def annual_albedo(station, min_coverage=MIN_JJA_COVERAGE):
    """Jaaralbedo als verhouding van de jaarsommen SWU/SWD (zoals paper fig. 9).

    Alleen jaren waarin SWD en SWU samen minstens min_coverage van de tijd gemeten zijn.
    """
    df = mc.load_aws(station)
    swd, swu = df["SWD"].clip(lower=0), df["SWU"].clip(lower=0)
    ok = swd.notna() & swu.notna()
    dt_h = df.index.to_series().diff().shift(-1).dt.total_seconds().div(3600).clip(upper=1.0)
    dt_h = dt_h.fillna(dt_h.median())
    g = df.index.year
    hours = pd.Series(df.index.year, index=df.index).groupby(g).first().map(
        lambda y: 8784 if pd.Timestamp(year=y, month=1, day=1).is_leap_year else 8760)
    out = pd.DataFrame({
        "albedo": swu.where(ok).groupby(g).sum() / swd.where(ok).groupby(g).sum(),
        "coverage": (dt_h * ok).groupby(g).sum() / hours,
        "aws_type": df["ID"].groupby(g).mean().round(),
    })
    out.index.name = "year"
    out["station"] = station
    return out[out["coverage"] >= min_coverage]


def plot_annual_albedo(annual, out):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for st in mc.AWS_STATIONS:
        d = annual[annual["station"] == st]
        if d.empty:
            continue
        color, m = mc.COLORS[st], STATION_MARKERS[st]
        ms = 12 if m == "*" else 8
        # lijn onderbreken bij ontbrekende jaren
        full = d["albedo"].reindex(range(d.index.min(), d.index.max() + 1))
        ax.plot(full.index, full.values, color=color, lw=2, zorder=2)
        for typ, face in [(0, color), (1, "white")]:
            e = d[d["aws_type"] == typ]
            ax.plot(e.index, e["albedo"], ls="", marker=m, ms=ms, mfc=face, mec=color, mew=1.5, zorder=3)
    from matplotlib.lines import Line2D
    st_handles = [Line2D([], [], color=mc.COLORS[st], lw=2, marker=STATION_MARKERS[st],
                         ms=12 if STATION_MARKERS[st] == "*" else 8, label=st)
                  for st in mc.AWS_STATIONS if st in set(annual["station"])]
    type_handles = [Line2D([], [], ls="", marker="o", ms=8, mfc=NEUTRAL, mec=NEUTRAL, label="old AWS type (0)"),
                    Line2D([], [], ls="", marker="o", ms=8, mfc="white", mec=NEUTRAL, mew=1.5,
                           label="new AWS type (1)")]
    leg = ax.legend(handles=st_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.add_artist(leg)
    ax.legend(handles=type_handles, frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0))
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual albedo (ΣSWU / ΣSWD)")
    ax.set_title("Annual mean albedo per station (years with ≥80% radiation data)", fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)


def plot_albedo_climatology(clims, years, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for st, c_full in clims.items():
        c = c_full.loc[120:270]
        if "p25" in c_full.attrs:
            lo, hi = c_full.attrs["p25"].loc[120:270], c_full.attrs["p75"].loc[120:270]
            ax1.fill_between(c.index, lo.values, hi.values, color=mc.COLORS[st], alpha=0.15, lw=0)
        ax1.plot(c.index, c.values, lw=2, color=mc.COLORS[st], label=f"{st} ({c_full.attrs.get('period', '?')})")
    ax1.set_xlabel("Day of year")
    ax1.set_ylabel("Albedo")
    ax1.set_title("Climatological albedo (median over all years, 15-day smoothed)\n"
                  "band: 25th–75th percentile between years", fontsize=10)
    ax1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    mc.style(ax1)
    for st in mc.AWS_STATIONS:
        d = years[years["station"] == st]
        r = regress(d["albedo"].values, d["melt"].values)
        ax2.scatter(d["albedo"], d["melt"], s=70, color=mc.COLORS[st], edgecolor="white", lw=1,
                    label=f"{st}  R² = {r['r2']:.2f}  (n = {r['n']})", zorder=3)
        if np.isfinite(r["slope"]):
            x = np.linspace(d["albedo"].min(), d["albedo"].max(), 5)
            ax2.plot(x, r["intercept"] + r["slope"] * x, color=mc.COLORS[st], lw=2.5, zorder=2)
    ax2.set_xlabel("Mean JJA albedo")
    ax2.set_ylabel("JJA melt (mm w.e.)")
    ax2.set_title("Summer melt vs summer albedo (per year)")
    ax2.legend(frameon=False)
    style(ax2)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
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
    annual_alb = pd.concat([annual_albedo(st) for st in mc.AWS_STATIONS])
    annual_alb.round(3).to_csv(out / "annual_albedo.csv")
    plot_annual_albedo(annual_alb, out / "annual_albedo.png")
    print(f"\nOutput opgeslagen in {out}")


if __name__ == "__main__":
    main()
