"""
Vergelijking van smelt (melt) tussen de K-transect stations.

Per station wordt de smelt berekend uit de oppervlakte-energiebalans (SEB):

    M = SWnet + LWnet + SHF + LHF        (W/m2, positief = naar het oppervlak)

Smelt treedt alleen op als het oppervlak op het smeltpunt is (Ts >= 0 degC,
met een kleine tolerantie) en M > 0. De bodemwarmtestroom G wordt verwaarloosd.
Smelt in mm w.e. = M * dt / Lf  (1 kg/m2 = 1 mm w.e.).

Stations en databronnen
-----------------------
AWS5, AWS6, AWS9, AWS10 (data/AWSx/*.tab, PANGAEA)
    Straling gemeten. Turbulente fluxen (SHF, LHF) berekend met de bulk-methode
    (Monin-Obukhov similariteit) uit wind, temperatuur, RH, druk en de
    oppervlaktetemperatuur afgeleid uit LWU.
    Daarnaast de *gemeten* oppervlakteverlaging (draw wire 'Ice melt' voor
    AWS type 1, akoestische hoogtemeter 'Height rel' voor type 0).
S5, SHR (data/S5, data/SHR, eddy-covariance masten)
    SHF gemeten (sonic). De LHF-kolom is in de bestanden leeg, dus LHF wordt
    berekend uit ustar, psih, z0h en qv0 - qvs. De masten meten geen straling:
    SWD, SWU en LWD komen van AWS5 (S5 is dezelfde locatie). Voor SHR is dat een
    benadering (vooral het albedo kan afwijken). LWU volgt uit de eigen
    oppervlaktetemperatuur (ths).

Gebruik
-------
    python code/melt_comparison.py                       # dagelijks, alle stations
    python code/melt_comparison.py --freq M              # maandelijks
    python code/melt_comparison.py --freq Y --stations AWS5 AWS6 AWS9 AWS10
    python code/melt_comparison.py --freq D --start 2020-06-01 --end 2020-09-01 --common

--freq accepteert pandas period-aliassen: h (uur), D (dag), W (week), M (maand), Y (jaar).
Output (CSV + figuren) komt in output/ tenzij --out anders zegt.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Fysische constanten
SIGMA = 5.670374e-8     # Stefan-Boltzmann (W m-2 K-4)
KAPPA = 0.4             # von Karman
G = 9.81                # zwaartekracht (m s-2)
RD = 287.05             # gasconstante droge lucht (J kg-1 K-1)
CP = 1005.0             # soortelijke warmte lucht (J kg-1 K-1)
LF = 3.34e5             # smeltwarmte (J kg-1)
LV = 2.501e6            # verdampingswarmte (J kg-1)
LS = LV + LF            # sublimatiewarmte (J kg-1)
T0 = 273.15             # smeltpunt (K)
RHO_ICE = 900.0         # dichtheid ijs voor omrekenen gemeten verlaging (kg m-3)

# Instellingen SEB
T_MELT_TOL = 0.5        # Ts binnen 0.5 K van smeltpunt -> smeltend oppervlak
MIN_COVERAGE = 0.8      # minimale datadekking per periode, anders NaN
Z_DEFAULT = 3.0         # meethoogte (m) als die niet bekend is
LWU_MELT = SIGMA * T0**4  # uitgaande langgolvige straling van smeltend oppervlak

# Ruwheidslengtes per AWS (m). Ablatiezone (AWS5/6) is ruw ijs, AWS9/10 sneeuw.
Z0M = {"AWS5": 1e-2, "AWS6": 5e-3, "AWS9": 1e-3, "AWS10": 1e-3}
Z0H_RATIO = 0.1         # z0h = z0m * Z0H_RATIO

AWS_STATIONS = ["AWS5", "AWS6", "AWS9", "AWS10"]
FLUX_STATIONS = ["S5", "SHR"]
ALL_STATIONS = AWS_STATIONS + FLUX_STATIONS

# Vaste kleur per station (categorisch palet, vaste volgorde)
# Engelse namen van de tijdseenheden voor de figuurteksten
FREQ_NAMES = {"h": "hour", "D": "day", "W": "week", "M": "month", "Y": "year"}

COLORS = {"AWS5": "#2a78d6", "AWS6": "#eb6834", "AWS9": "#1baf7a",
          "AWS10": "#eda100", "S5": "#e87ba4", "SHR": "#008300"}


# ---------------------------------------------------------------------------
# Inlezen
# ---------------------------------------------------------------------------

def read_tab(path):
    """Lees een PANGAEA .tab bestand (header eindigt met '*/')."""
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if line.startswith("*/"):
                break
    df = pd.read_csv(path, sep="\t", skiprows=n + 1, index_col=0, parse_dates=True)
    return df.rename(columns=lambda c: c.split(" [")[0])


def load_aws(station):
    files = sorted((DATA / station).glob("*.tab"))
    df = pd.concat([read_tab(f) for f in files])
    df = df[~df.index.duplicated()].sort_index()
    if "Ice melt" not in df:
        df["Ice melt"] = np.nan
    return df


def load_flux(station):
    files = sorted((DATA / station).glob("*.txt"))
    df = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in files])
    df = df[~df.index.duplicated()].sort_index()
    return df.rename(columns=lambda c: c.split("(")[0])


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def q_from_rh(t_k, rh, p_pa):
    """Specifieke vochtigheid uit RH (t.o.v. water, zoals Vaisala HMP)."""
    tc = t_k - T0
    es = 611.2 * np.exp(17.62 * tc / (243.12 + tc))
    e = rh / 100.0 * es
    return 0.622 * e / (p_pa - 0.378 * e)


def q_sat_ice(t_k, p_pa):
    tc = t_k - T0
    es = 611.2 * np.exp(22.46 * tc / (272.62 + tc))
    return 0.622 * es / (p_pa - 0.378 * es)


def psi_m(zeta):
    """Stabiliteitscorrectie impuls: Holtslag & De Bruin (stabiel), Paulson (instabiel)."""
    zs = np.maximum(zeta, 0)
    stable = -(0.7 * zs + 0.75 * (zs - 5 / 0.35) * np.exp(-0.35 * zs) + 0.75 * 5 / 0.35)
    x = (1 - 16 * np.minimum(zeta, 0)) ** 0.25
    unstable = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
    return np.where(zeta >= 0, stable, unstable)


def psi_h(zeta):
    zs = np.maximum(zeta, 0)
    stable = -((1 + 2 / 3 * 0.7 * zs) ** 1.5 + 0.75 * (zs - 5 / 0.35) * np.exp(-0.35 * zs)
               + 0.75 * 5 / 0.35 - 1)
    x = (1 - 16 * np.minimum(zeta, 0)) ** 0.25
    unstable = 2 * np.log((1 + x**2) / 2)
    return np.where(zeta >= 0, stable, unstable)


def bulk_fluxes(u, t_air, q_air, t_s, q_s, p, z, z0m, z0h, n_iter=10):
    """SHF en LHF (W/m2, positief naar oppervlak) met Monin-Obukhov iteratie."""
    u = np.maximum(u, 0.1)
    rho = p / (RD * t_air)
    dth = (t_air + G / CP * z) - t_s       # potentiele temperatuur verschil
    dq = q_air - q_s
    lnm, lnh = np.log(z / z0m), np.log(z / z0h)
    zeta = np.zeros_like(u)
    for _ in range(n_iter):
        ustar = KAPPA * u / (lnm - psi_m(zeta) + psi_m(zeta * z0m / z))
        denh = lnh - psi_h(zeta) + psi_h(zeta * z0h / z)
        thstar = KAPPA * dth / denh
        qstar = KAPPA * dq / denh
        thv_star = thstar + 0.61 * t_air * qstar
        L = ustar**2 * t_air / (KAPPA * G * np.where(np.abs(thv_star) < 1e-6, 1e-6, thv_star))
        zeta = np.clip(z / L, -5, 10)
    lv = np.where(t_s >= T0, LV, LS)
    return rho * CP * ustar * thstar, rho * lv * ustar * qstar


def seb_melt(swd, swu, lwd, lwu, shf, lhf, t_s):
    """Smeltenergie (W/m2) en componenten; smelt alleen bij smeltend oppervlak."""
    melting = t_s >= T0 - T_MELT_TOL
    lwu_eff = np.where(melting, LWU_MELT, lwu)
    swnet = swd.clip(lower=0) - swu.clip(lower=0)
    lwnet = lwd - lwu_eff
    q = swnet + lwnet + shf + lhf
    m = np.where(melting, np.maximum(q, 0), 0.0)
    m = np.where(np.isnan(q), np.nan, m)
    return pd.DataFrame({"SWnet": swnet, "LWnet": lwnet, "SHF": shf, "LHF": lhf,
                         "M": m}, index=swd.index)


# ---------------------------------------------------------------------------
# Smelt per station (op originele tijdresolutie)
# ---------------------------------------------------------------------------

def melt_aws(station, df=None):
    """SEB-smelt voor een AWS; geef df mee om met aangepaste invoer te rekenen."""
    if df is None:
        df = load_aws(station)
    p = df["PPPP"] * 100
    t_air = df["TTT"] + T0
    # oppervlaktetemperatuur uit LWU (emissiviteit 1), niet boven smeltpunt
    t_s = np.minimum((df["LWU"].clip(lower=0) / SIGMA) ** 0.25, T0)
    # meethoogte: bij AWS type 1 is 'Height rel' de sensorhoogte
    z = df["Height rel"].where((df["ID"] == 1) & df["Height rel"].between(0.5, 6), Z_DEFAULT)
    z0m = Z0M[station]
    shf, lhf = bulk_fluxes(df["ff"].values, t_air.values, q_from_rh(t_air, df["RH"], p).values,
                           t_s.values, q_sat_ice(t_s, p).values, p.values, z.values,
                           z0m, z0m * Z0H_RATIO)
    out = seb_melt(df["SWD"], df["SWU"], df["LWD"], df["LWU"],
                   pd.Series(shf, df.index), pd.Series(lhf, df.index), t_s)
    out["obs_lowering"] = observed_lowering(df)
    return out


def observed_lowering(df, max_step=0.05):
    """Gemeten oppervlakteverlaging per tijdstap (m, positief = verlaging).

    Type 1: draw wire ('Ice melt' neemt af bij smelt).
    Type 0: akoestische hoogtemeter op paal ('Height rel' neemt toe bij smelt).
    Sprongen > max_step (herinstallatie, onderhoud) worden weggegooid.
    """
    d_wire = -df["Ice melt"].diff()
    d_sr = df["Height rel"].diff()
    d = d_wire.where(df["ID"] == 1, d_sr.where(df["ID"] == 0))
    return d.where(d.abs() <= max_step)


def melt_flux(station):
    df = load_flux(station)
    rad = load_aws("AWS5")[["SWD", "SWU", "LWD"]].reindex(df.index)
    p = df["p0"]
    t_s = np.minimum(df["ths"] * (p / 1e5) ** (RD / CP), T0)
    shf = df["SHF_sonic"].fillna(df["SHF_couple"])
    # LHF uit flux-profiel relatie: LHF = rho Lv kappa u* (q0 - qs) / (ln(z/z0h) - psih)
    z0h = df["z0h"].where(df["z0h"] > 0).fillna(df["z0h"][df["z0h"] > 0].median())
    lv = np.where(t_s >= T0, LV, LS)
    lhf = (df["rhoa"] * lv * KAPPA * df["ustar"] * (df["qv0"] - df["qvs"])
           / (np.log(df["z"] / z0h) - df["psih"]))
    # Albedo/SWU van AWS5 schalen we niet; zie docstring bovenaan.
    out = seb_melt(rad["SWD"], rad["SWU"], rad["LWD"], SIGMA * t_s**4, shf, lhf, t_s)
    out["obs_lowering"] = np.nan
    return out


def melt_station(station):
    return melt_aws(station) if station in AWS_STATIONS else melt_flux(station)


# ---------------------------------------------------------------------------
# Aggregeren per tijdseenheid
# ---------------------------------------------------------------------------

def aggregate(ts, freq, min_coverage=MIN_COVERAGE):
    """Smelt (mm w.e.) en gemiddelde fluxen per periode, met dekkingscontrole."""
    ts = ts.copy()
    # tijdstap per rij (h); grote gaten tellen niet als dekking
    dt_h = ts.index.to_series().diff().shift(-1).dt.total_seconds() / 3600
    ts["dt_h"] = dt_h.fillna(dt_h.median()).clip(upper=1.0)
    ts["melt_rate"] = ts["M"] * 3600 / LF                     # mm w.e. per uur
    periods = ts.index.to_period(freq)
    hours = ((periods.end_time - periods.start_time).total_seconds() + 1e-9) / 3600

    valid = ts["M"].notna()
    g = ts.groupby(periods)
    res = pd.DataFrame({
        "coverage": (ts["dt_h"] * valid).groupby(periods).sum() / pd.Series(hours, ts.index).groupby(periods).first(),
        "SWnet": g["SWnet"].mean(), "LWnet": g["LWnet"].mean(),
        "SHF": g["SHF"].mean(), "LHF": g["LHF"].mean(), "M": g["M"].mean(),
    })
    res["melt_mmwe"] = g["melt_rate"].mean() * pd.Series(hours, ts.index).groupby(periods).first()
    res["melt_hours"] = (ts["dt_h"] * (ts["M"] > 0)).groupby(periods).sum()
    obs = ts["obs_lowering"]
    obs_cov = (ts["dt_h"] * obs.notna()).groupby(periods).sum() / pd.Series(hours, ts.index).groupby(periods).first()
    res["obs_lowering_m"] = obs.groupby(periods).sum().where(obs_cov >= min_coverage)
    res["obs_melt_mmwe_ice"] = res["obs_lowering_m"] * RHO_ICE   # alleen zinvol voor kaal ijs
    bad = res["coverage"] < min_coverage
    res.loc[bad, ["SWnet", "LWnet", "SHF", "LHF", "M", "melt_mmwe"]] = np.nan
    res["coverage"] = res["coverage"].clip(upper=1).round(3)
    res.index.name = "period"
    return res


# ---------------------------------------------------------------------------
# Figuren
# ---------------------------------------------------------------------------

def style(ax):
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#a8a7a2")
    ax.tick_params(colors="#52514e")


def plot_timeseries(wide, freq, out):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = wide.index.to_timestamp()
    for st in wide.columns:
        y = wide[st]
        if freq in ("h", "D", "W"):
            ax.plot(x, y, lw=2, color=COLORS[st], label=st)
        else:
            ax.plot(x, y, lw=2, marker="o", ms=4, color=COLORS[st], label=st)
    ax.set_ylabel(f"melt (mm w.e. per {FREQ_NAMES.get(freq, freq)})")
    ax.set_title("Calculated melt per station", loc="left", color="#0b0b0b")
    ax.legend(frameon=False, ncol=len(wide.columns), loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_components(long, out):
    """Gemiddelde SEB-componenten tijdens perioden met smelt, per station."""
    comp = ["SWnet", "LWnet", "SHF", "LHF"]
    ccol = {"SWnet": "#2a78d6", "LWnet": "#eb6834", "SHF": "#1baf7a", "LHF": "#eda100"}
    sub = long[long["melt_mmwe"] > 0].groupby("station")[comp].mean()
    sub = sub.reindex([s for s in ALL_STATIONS if s in sub.index])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(comp)
    xs = np.arange(len(sub))
    for i, c in enumerate(comp):
        ax.bar(xs + (i - 1.5) * w, sub[c], width=w * 0.92, color=ccol[c], label=c)
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(xs, sub.index)
    ax.set_ylabel("W m$^{-2}$")
    ax.set_title("Mean surface energy balance components during melt periods", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_calc_vs_obs(long, freq, out):
    sub = long.dropna(subset=["melt_mmwe", "obs_melt_mmwe_ice"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for st in [s for s in ALL_STATIONS if s in set(sub["station"])]:
        d = sub[sub["station"] == st]
        ax.scatter(d["obs_melt_mmwe_ice"], d["melt_mmwe"], s=18, alpha=0.7,
                   color=COLORS[st], edgecolor="white", lw=0.5, label=st)
    lim = [0, max(sub["melt_mmwe"].max(), sub["obs_melt_mmwe_ice"].max()) * 1.05]
    ax.plot(lim, lim, color="#a8a7a2", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"observed (mm w.e. per {FREQ_NAMES.get(freq, freq)}, assuming ice {RHO_ICE:.0f} kg/m³)")
    ax.set_ylabel(f"calculated SEB (mm w.e. per {FREQ_NAMES.get(freq, freq)})")
    ax.set_title("Calculated vs observed melt", loc="left")
    ax.legend(frameon=False)
    style(ax)
    ax.grid(axis="x", color="#e5e4e0", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", default="D", help="tijdseenheid: h, D, W, M, Y (default D)")
    ap.add_argument("--stations", nargs="+", default=ALL_STATIONS, choices=ALL_STATIONS)
    ap.add_argument("--start", help="begindatum, bv. 2019-06-01")
    ap.add_argument("--end", help="einddatum (exclusief), bv. 2019-09-01")
    ap.add_argument("--common", action="store_true",
                    help="alleen perioden waarin alle gekozen stations data hebben")
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parts = []
    for st in args.stations:
        print(f"{st}: smelt berekenen ...")
        ts = melt_station(st)
        if args.start or args.end:
            ts = ts.loc[args.start:args.end]
            if args.end:
                ts = ts[ts.index < pd.Timestamp(args.end)]
        res = aggregate(ts, args.freq)
        res.insert(0, "station", st)
        parts.append(res)
    long = pd.concat(parts)

    wide = long.reset_index().pivot(index="period", columns="station", values="melt_mmwe")
    wide = wide[[s for s in args.stations if s in wide.columns]]
    if args.common:
        wide = wide.dropna()
        long = long[long.index.isin(wide.index)]
    else:
        wide = wide.dropna(how="all")

    tag = f"{args.freq}"
    long.to_csv(out / f"melt_per_station_{tag}.csv")
    wide.to_csv(out / f"melt_comparison_{tag}.csv")

    # samenvatting per station
    summary = long.groupby("station").agg(
        perioden=("melt_mmwe", "count"),
        totaal_smelt_mmwe=("melt_mmwe", "sum"),
        gemiddeld_per_periode=("melt_mmwe", "mean"),
        max_per_periode=("melt_mmwe", "max"),
        smelturen=("melt_hours", "sum"),
    ).reindex(args.stations).round(1)
    summary.to_csv(out / f"melt_summary_{tag}.csv")
    print(f"\nSmelt per station (tijdseenheid {args.freq}):")
    print(summary.to_string())

    plot_timeseries(wide, args.freq, out / f"melt_timeseries_{tag}.png")
    plot_components(long, out / f"seb_components_{tag}.png")
    plot_calc_vs_obs(long, args.freq, out / f"melt_calc_vs_obs_{tag}.png")
    print(f"\nOutput opgeslagen in {out}")


if __name__ == "__main__":
    main()
