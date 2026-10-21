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
    oppervlaktetemperatuur afgeleid uit LWU. Ruwheidslengtes zoals in
    Kuipers Munneke et al. (2018); zie Z0M en SCALAR_SCHEME.
    Daarnaast de *gemeten* oppervlakteverlaging (draw wire 'Ice melt' voor
    AWS type 1, akoestische hoogtemeter 'Height rel' voor type 0).
SHR (data/SHR, eddy-covariance mast)
    SHF gemeten (sonic). De LHF-kolom is in het bestand leeg, dus LHF wordt
    berekend uit ustar, psih, z0h en qv0 - qvs. De mast meet geen straling:
    SWD, SWU en LWD komen van AWS5, wat een benadering is (vooral het albedo
    kan afwijken). LWU volgt uit de eigen oppervlaktetemperatuur (ths).
S5 (data/S5) is dezelfde locatie als AWS5 en wordt daarom niet meegenomen.

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

from matplotlib.lines import Line2D

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
RHO_ICE = 910.0         # dichtheid ijs voor omrekenen gemeten verlaging (kg m-3)

# Instellingen SEB
T_MELT_TOL = 0.5        # Ts binnen 0.5 K van smeltpunt -> smeltend oppervlak
MIN_COVERAGE = 0.8      # minimale datadekking per periode, anders NaN
Z_DEFAULT = 3.0         # meethoogte (m) als die niet bekend is
LWU_MELT = SIGMA * T0**4  # uitgaande langgolvige straling van smeltend oppervlak

# Ruwheidslengtes per AWS (m). Ablatiezone (AWS5/6) is ruw ijs, AWS9/10 sneeuw.
# Ruwheid zoals Kuipers Munneke et al. (2018):
# - S5/S6: in de tijd varierende z0m (20-daags lopend gemiddelde). De AWS-data hebben maar
#   een meetniveau, dus z0m komt uit de eddy-covariance masten: S5 (zelfde locatie als AWS5)
#   en, als benadering voor AWS6 (geen eigen mast), SHR. Waar geen mastdata zijn wordt de
#   seizoensklimatologie (per dag van het jaar) gebruikt.
# - S9/S10: constante z0m = 0.126 mm.
# - zh en zq: Andreas (1987); bij AWS5 voor z0m > 1 mm Smeets & van den Broeke (2008).
Z0M = {"AWS5": "S5", "AWS6": "SHR", "AWS9": 1.26e-4, "AWS10": 1.26e-4}   # waarde (m) of bronmast
SCALAR_SCHEME = {"AWS5": "smeets2008", "AWS6": "andreas1987",
                 "AWS9": "andreas1987", "AWS10": "andreas1987"}
Z0M_WINDOW_DAYS = 20     # lopend gemiddelde zoals in het paper
Z0M_VALID = (1e-5, 0.5)  # bruikbare z0m-waarden uit de fluxbestanden (m)
NU_AIR = 1.46e-5         # kinematische viscositeit lucht (m2/s) voor R* = u* z0 / nu

AWS_STATIONS = ["AWS5", "AWS6", "AWS9", "AWS10"]
FLUX_STATIONS = ["SHR"]   # S5 = zelfde locatie als AWS5, daarom niet meegenomen
ALL_STATIONS = AWS_STATIONS + FLUX_STATIONS

# Vaste kleur per station (categorisch palet, vaste volgorde)
# Engelse namen van de tijdseenheden voor de figuurteksten
FREQ_NAMES = {"h": "hour", "D": "day", "W": "week", "M": "month", "Y": "year"}

COLORS = {"AWS5": "#0e8a8a",  
          "AWS6": "#e5a50a",   
          "AWS9": "#6aa121",   
          "AWS10": "#7b3fa0",  
          "SHR": "#d6453d"}   


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


def scalar_roughness(ustar, z0m, scheme):
    """Ruwheidslengtes voor warmte en vocht (z0h, z0q) uit R* = u* z0m / nu.

    andreas1987: ln(zs/z0) = b0 + b1 ln R* + b2 (ln R*)^2, coefficienten per regime
                 (glad R* <= 0.135, overgang < 2.5, ruw >= 2.5).
    smeets2008:  zelfde vorm, b = (1.5, -0.2, -0.11) voor warmte en (1.5, -0.2, -0.16)
                 voor vocht; alleen voor z0m > 1 mm, anders andreas1987.
    """
    rstar = np.clip(ustar * z0m / NU_AIR, 1e-3, 1e3)
    lr = np.log(rstar)
    # (b0, b1, b2) voor warmte en vocht, Andreas (1987) tabel 1
    smooth, trans = rstar <= 0.135, (rstar > 0.135) & (rstar < 2.5)
    def andreas(c_smooth, c_trans, c_rough):
        b = [np.select([smooth, trans], [c_smooth[i], c_trans[i]], c_rough[i]) for i in range(3)]
        return b[0] + b[1] * lr + b[2] * lr**2
    ln_h = andreas((1.250, 0, 0), (0.149, -0.550, 0), (0.317, -0.565, -0.183))
    ln_q = andreas((1.610, 0, 0), (0.351, -0.628, 0), (0.396, -0.512, -0.180))
    if scheme == "smeets2008":
        rough = z0m > 1e-3
        ln_h = np.where(rough, 1.5 - 0.2 * lr - 0.11 * lr**2, ln_h)
        ln_q = np.where(rough, 1.5 - 0.2 * lr - 0.16 * lr**2, ln_q)
    return z0m * np.exp(ln_h), z0m * np.exp(ln_q)


def bulk_fluxes(u, t_air, q_air, t_s, q_s, p, z, z0m, scheme="andreas1987", n_iter=10):
    """SHF en LHF (W/m2, positief naar oppervlak) met Monin-Obukhov iteratie.

    z0m mag per tijdstap verschillen; z0h en z0q worden elke iteratie opnieuw
    bepaald uit u* (zie scalar_roughness).
    """
    u = np.maximum(u, 0.1)
    z0m = np.broadcast_to(z0m, u.shape)
    rho = p / (RD * t_air)
    dth = (t_air + G / CP * z) - t_s       # potentiele temperatuur verschil
    dq = q_air - q_s
    lnm = np.log(z / z0m)
    zeta = np.zeros_like(u)
    ustar = KAPPA * u / lnm                # neutrale startwaarde
    for _ in range(n_iter):
        z0h, z0q = scalar_roughness(ustar, z0m, scheme)
        ustar = KAPPA * u / (lnm - psi_m(zeta) + psi_m(zeta * z0m / z))
        denh = np.log(z / z0h) - psi_h(zeta) + psi_h(zeta * z0h / z)
        denq = np.log(z / z0q) - psi_h(zeta) + psi_h(zeta * z0q / z)
        thstar = KAPPA * dth / denh
        qstar = KAPPA * dq / denq
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

_Z0M_CACHE = {}


def z0m_series(station, index):
    """z0m (m) per tijdstap: constant, of uit de eddy-covariance mast (zie Z0M).

    Uit de mastdata: daggemiddelde (mediaan) -> 20-daags lopend gemiddelde.
    Dagen zonder mastdata krijgen de klimatologie voor die dag van het jaar.
    """
    src = Z0M[station]
    if not isinstance(src, str):
        return np.full(len(index), float(src))
    if src not in _Z0M_CACHE:
        z = load_flux(src)["z0m"]
        z = z.where(z.between(*Z0M_VALID))
        daily = z.resample("D").median()
        running = daily.rolling(Z0M_WINDOW_DAYS, center=True, min_periods=5).mean()
        clim = running.groupby(running.index.dayofyear).median().reindex(range(1, 367))
        # cyclisch gladstrijken en gaten in de klimatologie dichten
        wrap = pd.concat([clim.iloc[-Z0M_WINDOW_DAYS:], clim, clim.iloc[:Z0M_WINDOW_DAYS]])
        wrap = wrap.interpolate().rolling(Z0M_WINDOW_DAYS, center=True, min_periods=1).mean()
        clim = wrap.iloc[Z0M_WINDOW_DAYS:-Z0M_WINDOW_DAYS]
        _Z0M_CACHE[src] = (running, clim)
    running, clim = _Z0M_CACHE[src]
    days = index.floor("D")
    actual = running.reindex(days).values
    fallback = clim.reindex(index.dayofyear).values
    return np.where(np.isnan(actual), fallback, actual)


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
    z0m = z0m_series(station, df.index)
    shf, lhf = bulk_fluxes(df["ff"].values, t_air.values, q_from_rh(t_air, df["RH"], p).values,
                           t_s.values, q_sat_ice(t_s, p).values, p.values, z.values,
                           z0m, SCALAR_SCHEME[station])
    out = seb_melt(df["SWD"], df["SWU"], df["LWD"], df["LWU"],
                   pd.Series(shf, df.index), pd.Series(lhf, df.index), t_s)
    out["obs_lowering"], out["obs_dt_h"] = observed_lowering(df)
    return out


def observed_lowering(df, max_step=0.05):
    """Gemeten oppervlakteverlaging per tijdstap (m, positief = verlaging).

    Type 1: draw wire ('Ice melt' neemt af bij smelt).
    Type 0: akoestische hoogtemeter op paal ('Height rel' neemt toe bij smelt).
    Sprongen > max_step (herinstallatie, onderhoud) worden weggegooid.
    Geeft ook de duur (h) van elke geldige stap terug, zodat aggregate() de
    gemeten verlaging net als de berekende smelt kan opschalen naar de periode.
    """
    d_wire = -df["Ice melt"].diff()
    d_sr = df["Height rel"].diff()
    d = d_wire.where(df["ID"] == 1, d_sr.where(df["ID"] == 0))
    d = d.where(d.abs() <= max_step)
    dt_h = df.index.to_series().diff().dt.total_seconds() / 3600
    return d, dt_h.where(d.notna())


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
    out["obs_dt_h"] = np.nan
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
    period_hours = pd.Series(hours, ts.index).groupby(periods).first()

    valid = ts["M"].notna()
    g = ts.groupby(periods)
    res = pd.DataFrame({
        "coverage": (ts["dt_h"] * valid).groupby(periods).sum() / period_hours,
        "SWnet": g["SWnet"].mean(), "LWnet": g["LWnet"].mean(),
        "SHF": g["SHF"].mean(), "LHF": g["LHF"].mean(), "M": g["M"].mean(),
    })
    res["melt_mmwe"] = g["melt_rate"].mean() * period_hours
    res["melt_hours"] = (ts["dt_h"] * (ts["M"] > 0)).groupby(periods).sum()
    # gemeten verlaging: net als de berekende smelt opschalen van de gedekte tijd naar de hele periode
    obs_hours = ts["obs_dt_h"].groupby(periods).sum()
    res["obs_coverage"] = (obs_hours / period_hours).clip(upper=1).round(3)
    obs_sum = ts["obs_lowering"].groupby(periods).sum()
    res["obs_lowering_m"] = (obs_sum / obs_hours * period_hours).where(res["obs_coverage"] >= min_coverage)
    res["obs_melt_mmwe_ice"] = res["obs_lowering_m"] * RHO_ICE   # alleen zinvol voor kaal ijs
    bad = res["coverage"] < min_coverage
    res.loc[bad, ["SWnet", "LWnet", "SHF", "LHF", "M", "melt_mmwe"]] = np.nan
    res["coverage"] = res["coverage"].clip(upper=1).round(3)
    res.index.name = "period"
    return res


# ---------------------------------------------------------------------------
# Figuren
# ---------------------------------------------------------------------------

# def style(ax):
#     ax.grid(axis="y", color="#e5e4e0", lw=0.8)
#     ax.set_axisbelow(True)
#     for s in ("top", "right"):
#         ax.spines[s].set_visible(False)
#     for s in ("left", "bottom"):
#         ax.spines[s].set_color("#a8a7a2")
#     ax.tick_params(colors="#52514e")

def style(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("black")
        s.set_linewidth(1)
    ax.grid(color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)                       
    ax.tick_params(direction="out", length=5, width=1, colors="black")

def plot_timeseries(wide, freq, out):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = wide.index.to_timestamp()
    use_markers = freq not in ("h", "D", "W")
    for st in wide.columns:
        y = wide[st]
        if use_markers:
            ax.plot(x, y, lw=2, marker="o", ms=4, color=COLORS[st])
        else:
            ax.plot(x, y, lw=2, color=COLORS[st])
    ax.set_ylabel(f"Melt (mm w.e. per {FREQ_NAMES.get(freq, freq)})")
    ax.set_xlabel("Time (year)")
    ax.set_title("Calculated melt per station", color="#0b0b0b")

    handles = [Line2D([], [], lw=3, color=COLORS[st], label=st,
                      marker="o" if use_markers else None, markersize=5)
               for st in wide.columns]
    ax.legend(handles=handles, loc="upper left", ncol=len(wide.columns),
              frameon=True, facecolor="white", edgecolor="#a8a7a2",
              framealpha=1, borderpad=0.8, columnspacing=1.5)

    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
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
    ax.set_title("Mean surface energy balance components during melt periods")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)

def plot_calc_vs_obs(long, freq, out):
    sub = long.dropna(subset=["melt_mmwe", "obs_melt_mmwe_ice"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    stations = [s for s in ALL_STATIONS if s in set(sub["station"])]
    for st in stations:
        d = sub[sub["station"] == st]
        ax.scatter(d["obs_melt_mmwe_ice"], d["melt_mmwe"], s=30, alpha=0.75,
                   color=COLORS[st], edgecolor="none")
    lim = [0, max(sub["melt_mmwe"].max(), sub["obs_melt_mmwe_ice"].max()) * 1.05]
    ax.plot(lim, lim, color="#a8a7a2", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observed (mm w.e. per {FREQ_NAMES.get(freq, freq)})")
    ax.set_ylabel(f"Calculated SEB (mm w.e. per {FREQ_NAMES.get(freq, freq)})")
    ax.set_title("Calculated vs observed melt")

    handles = [Line2D([], [], marker="o", ls="", markersize=8,
                      color=COLORS[st], label=st) for st in stations]
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white",
              edgecolor="#a8a7a2", framealpha=1, borderpad=0.8, labelspacing=0.6)

    style(ax)
    ax.grid(axis="x", color="#e5e4e0", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", default="Y", help="tijdseenheid: h, D, W, M, Y (default D)")
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

    stake = {   # net surface mass balance (m w.e.), mass balance year ending late August
    "AWS5": {2004: -3.99, 2005: -3.63, 2006: -3.70, 2007: -4.19, 2008: -4.10, 2009: -3.38, 2010: -5.12},
    "AWS6": {2004: -1.47, 2005: -1.28, 2006: -1.50, 2007: -2.00, 2008: -1.48, 2009: -1.34, 2010: -2.95},
}

rows = []
for st in ["AWS5", "AWS6", "AWS9", "AWS10"]:
    ts = melt_station(st)
    s = ts[ts.index.month.isin([6, 7, 8])]
    g = s.groupby(s.index.year)
    jja = g[["SWnet", "LWnet", "SHF", "LHF", "M"]].mean()
    jja["Rnet"] = jja["SWnet"] + jja["LWnet"]
    jja = jja[s["M"].notna().groupby(s.index.year).mean() >= 0.8]    # enough data
    jja = jja[(jja.index >= 2004) & (jja.index <= 2016)]    # same summers as the pape
    r = jja.corr()["M"][["SWnet", "LWnet", "Rnet", "SHF", "LHF"]].round(2)
    r.name = f"{st} (n={len(jja)})"
    rows.append(r)
print(pd.DataFrame(rows).T)

rows = {}
for st in ["AWS5", "AWS6", "AWS9"]:
    df = load_aws(st)
    j = df[df.index.month.isin([6, 7, 8])]
    g = j.groupby(j.index.year)
    rows[(st, "SWD")] = g["SWD"].mean()
    rows[(st, "albedo")] = g["SWU"].sum() / g["SWD"].sum()
tab = pd.DataFrame(rows).round(2)
print(tab)