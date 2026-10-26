"""
Gemodelleerde tegen gemeten oppervlaktetemperatuur (zoals fig. 2 van Kuipers Munneke et al., 2018).

Gemodelleerd: per tijdstap de oppervlaktetemperatuur Ts waarbij de energiebalans sluit,

    SWnet + LWD - sigma*Ts^4 + SHF(Ts) + LHF(Ts) + G = 0,

met Ts begrensd op 0 degC (dan smelt het oppervlak). SHF en LHF komen uit dezelfde
bulkmethode en ruwheid als in melt_comparison.py; G komt uit het ondergrondmodel hieronder.
De balans neemt monotoon af met Ts, dus de oplossing wordt met bisectie gevonden.

Gemeten: Ts uit de gemeten LWU (emissiviteit 1).

G komt uit een eenvoudig ondergrondmodel: 1D warmtegeleiding in 20 m ijs of firn
(impliciet, ~40 lagen, bovenste laag 4 cm), met Ts als bovenrand en een vaste
temperatuur onderin. Ts en G worden gekoppeld door herhaald:
  1) Ts oplossen met G gelineariseerd rond de vorige iteratie: G(Ts) = G_ref - K_eff (Ts - Ts_ref),
     met K_eff de directe reactie van de bovenste laag binnen een tijdstap,
  2) het ondergrondmodel draaien met die Ts -> nieuwe G_ref, en herhalen tot het stabiel is.
  Zonder die linearisering is de koppeling in ijs (K_eff ~ 100 W/m2/K) instabiel.
Ook wordt het resultaat zonder G (G = 0) bewaard ter vergelijking.
Smeltwater, percolatie en hervriezen zitten er niet in (ijs wordt begrensd op 0 degC).

Gebruik
-------
    python code/surface_temperature.py [--out output/surface_temperature]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.linalg import lu_factor, lu_solve

import melt_comparison as mc

TS_MIN = 200.0          # ondergrens zoekinterval (K)
N_BISECT = 20           # stappen bisectie (interval wordt < 1e-4 K)
N_G_ITER = 6            # koppelingsiteraties Ts <-> G
G_DAMPING = 1.0         # nieuwe G = demping * G_model + (1 - demping) * G_vorige (1 = geen demping)

# Ondergrond per station (aanname): dichtheid (kg/m3) en warmtegeleiding (W/m/K).
# AWS5/AWS6 kaal ijs; AWS9/AWS10 firn, geleiding volgens Anderson (1976): 0.021 + 2.5 (rho/1000)^2
SUBSURFACE = {"AWS5": (900.0, 2.22), "AWS6": (900.0, 2.22),
              "AWS9": (500.0, 0.021 + 2.5 * 0.5**2), "AWS10": (450.0, 0.021 + 2.5 * 0.45**2)}
C_ICE = 2090.0          # warmtecapaciteit ijs (J/kg/K)
DEPTH = 20.0            # diepte ondergrondmodel (m), zoals het paper
DZ_TOP, DZ_GROWTH = 0.04, 1.15


def energy_balance(ts, d):
    """Netto energiebalans (W/m2) bij oppervlaktetemperatuur ts; d bevat de invoer."""
    shf, lhf = mc.bulk_fluxes(d["u"], d["t_air"], d["q_air"], ts, mc.q_sat_ice(ts, d["p"]),
                              d["p"], d["z"], d["z0m"], d["scheme"], n_iter=6)
    # G gelineariseerd rond de vorige iteratie: G(Ts) = G_ref - K_eff * (Ts - Ts_ref)
    g = d["g_ref"] - d["k_eff"] * (ts - d["ts_ref"])
    return d["swnet"] + d["lwd"] - mc.SIGMA * ts**4 + shf + lhf + g


def hourly_input(station, df=None):
    """Uurgemiddelde invoer voor de energiebalans (df: eventueel aangepaste AWS-data)."""
    df = (mc.load_aws(station) if df is None else df).resample("h").mean()
    df["ID"] = df["ID"].round()
    p = (df["PPPP"] * 100).values
    t_air = (df["TTT"] + mc.T0).values
    z = df["Height rel"].where((df["ID"] == 1) & df["Height rel"].between(0.5, 6), mc.Z_DEFAULT).values
    d = {"u": df["ff"].values, "t_air": t_air, "p": p, "z": z,
         "q_air": mc.q_from_rh(pd.Series(t_air), pd.Series(df["RH"].values), pd.Series(p)).values,
         "z0m": mc.z0m_series(station, df.index), "scheme": mc.SCALAR_SCHEME[station],
         "swnet": (df["SWD"].clip(lower=0) - df["SWU"].clip(lower=0)).values,
         "lwd": df["LWD"].values,
         "g_ref": np.zeros(len(df)), "ts_ref": np.full(len(df), mc.T0), "k_eff": 0.0}
    ts_obs = ((df["LWU"].clip(lower=0) / mc.SIGMA) ** 0.25).values - mc.T0
    return df.index, d, ts_obs


def solve_ts(d):
    """Ts (K) waarbij de energiebalans sluit, begrensd op 0 degC."""
    n = len(d["u"])
    hi = np.full(n, mc.T0)
    melting = energy_balance(hi, d) >= 0          # bij 0 degC nog energie over -> smelt
    lo = np.full(n, TS_MIN)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        pos = energy_balance(mid, d) > 0          # nog energie over -> Ts ligt hoger
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    ts = np.where(melting, mc.T0, 0.5 * (lo + hi))
    valid = np.isfinite(energy_balance(np.full(n, 260.0), d))
    return np.where(valid, ts, np.nan)


def subsurface_grid():
    dz = []
    while sum(dz) < DEPTH:
        dz.append(DZ_TOP * DZ_GROWTH ** len(dz))
    return np.array(dz)


def ground_heat_flux(ts_c, station, dt_s=3600.0):
    """G (W/m2, + = naar het oppervlak) uit 1D warmtegeleiding met Ts (degC) als bovenrand."""
    rho, k = SUBSURFACE[station]
    dz = subsurface_grid()
    n = len(dz)
    cap = rho * C_ICE * dz / dt_s                         # J/m2/K per stap
    k_mid = k / (0.5 * (dz[:-1] + dz[1:]))                # geleiding tussen lagen
    k_top, k_bot = k / (0.5 * dz[0]), k / (0.5 * dz[-1])
    a = np.diag(cap)
    a[np.arange(n - 1), np.arange(1, n)] = -k_mid
    a[np.arange(1, n), np.arange(n - 1)] = -k_mid
    a[np.arange(n), np.arange(n)] += np.r_[k_top, k_mid] + np.r_[k_mid, k_bot]
    lu = lu_factor(a)
    # directe reactie van G op Ts binnen een stap: G = K_top (T_0 - Ts), dT_0/dTs = beta
    e0 = np.zeros(n)
    e0[0] = k_top
    beta = lu_solve(lu, e0)[0]
    k_eff = k_top * (1 - beta)

    ts = pd.Series(ts_c).ffill().bfill().values           # gaten: vorige waarde vasthouden
    t_bot = min(np.nanmean(ts), 0.0)                       # vaste temperatuur onderin
    temp = np.full(n, t_bot)

    def step(temp, t_surf):
        rhs = cap * temp
        rhs[0] += k_top * t_surf
        rhs[-1] += k_bot * t_bot
        return np.minimum(lu_solve(lu, rhs, check_finite=False), 0.0)

    for t_surf in ts[:min(len(ts), 8760)]:                # inspinnen: eerste jaar
        temp = step(temp, t_surf)
    g = np.empty(len(ts))
    for i, t_surf in enumerate(ts):
        temp = step(temp, t_surf)
        g[i] = k_top * (temp[0] - t_surf)
    return g, k_eff


def coupled(station, df=None, verbose=True):
    """Gekoppelde oplossing van Ts en G; geeft index, invoer d, Ts (degC), gemeten Ts en Ts bij G = 0."""
    index, d, ts_obs = hourly_input(station, df)
    ts0 = solve_ts(d) - mc.T0
    ts = ts0
    for it in range(N_G_ITER):
        g_new, k_eff = ground_heat_flux(ts, station)
        d["g_ref"] = G_DAMPING * g_new + (1 - G_DAMPING) * d["g_ref"]
        d["ts_ref"] = pd.Series(ts).ffill().bfill().values + mc.T0
        d["k_eff"] = k_eff
        ts_prev, ts = ts, solve_ts(d) - mc.T0
        ok = np.isfinite(ts) & np.isfinite(ts_obs)
        if verbose:
            print(f"   iteratie {it + 1}: RMSE {np.sqrt(np.mean((ts[ok] - ts_obs[ok])**2)):.2f} °C, "
              f"gem. verandering Ts {np.nanmean(np.abs(ts - ts_prev)):.3f} °C (K_eff {k_eff:.0f} W/m²/K)",
              flush=True)
    return index, d, ts, ts_obs, ts0


def modelled_ts(station):
    """Gemodelleerde Ts zonder G en met gekoppelde G, plus gemeten Ts (degC), per uur."""
    index, d, ts, ts_obs, ts0 = coupled(station)
    g = d["g_ref"] - d["k_eff"] * (ts + mc.T0 - d["ts_ref"])
    return pd.DataFrame({"ts_mod": ts, "ts_mod_g0": ts0, "ts_obs": ts_obs, "g": g}, index=index)


def melt_coupled(station, df=None):
    """Smelt zoals in het paper: restterm van de energiebalans als de berekende Ts = 0 degC.

    Geeft uurwaarden met dezelfde kolommen als melt_comparison.melt_aws, plus G en Ts_mod.
    """
    native = mc.load_aws(station) if df is None else df
    index, d, ts_c, _, _ = coupled(station, native, verbose=False)
    ts = ts_c + mc.T0
    shf, lhf = mc.bulk_fluxes(d["u"], d["t_air"], d["q_air"], ts, mc.q_sat_ice(ts, d["p"]),
                              d["p"], d["z"], d["z0m"], d["scheme"])
    g = d["g_ref"] - d["k_eff"] * (ts - d["ts_ref"])
    lwnet = d["lwd"] - mc.SIGMA * ts**4
    q = d["swnet"] + lwnet + shf + lhf + g
    m = np.where(ts >= mc.T0 - 1e-6, np.maximum(q, 0), 0.0)
    m = np.where(np.isfinite(q) & np.isfinite(ts), m, np.nan)
    out = pd.DataFrame({"SWnet": d["swnet"], "LWnet": lwnet, "SHF": shf, "LHF": lhf, "G": g,
                        "M": m, "Ts_mod": ts_c}, index=index)
    low, dt = mc.observed_lowering(native)
    out["obs_lowering"] = low.resample("h").sum(min_count=1).reindex(index)
    out["obs_dt_h"] = dt.resample("h").sum(min_count=1).reindex(index)
    return out


_MELT_CACHE = {}


def melt_coupled_cached(station, df=None):
    """melt_coupled met cache (geheugen + schijf) voor de standaardinvoer.

    De schijfcache hoort bij een hash van alles wat de berekening bepaalt (dit bestand, de
    natuurkundige functies en instellingen in melt_comparison.py), zodat hij vervalt als de
    berekening verandert, maar niet bij bijvoorbeeld andere plotkleuren.
    """
    if df is not None:
        return melt_coupled(station, df)
    import hashlib
    import inspect
    funcs = [mc.read_tab, mc.load_aws, mc.load_flux, mc.q_from_rh, mc.q_sat_ice, mc.psi_m, mc.psi_h,
             mc.scalar_roughness, mc.bulk_fluxes, mc.z0m_series, mc.observed_lowering]
    consts = [mc.Z0M, mc.SCALAR_SCHEME, mc.Z0M_WINDOW_DAYS, mc.Z0M_VALID, mc.NU_AIR, mc.Z_DEFAULT,
              mc.SIGMA, mc.KAPPA, mc.G, mc.RD, mc.CP, mc.LF, mc.LV, mc.LS, mc.T0]
    src = (Path(__file__).read_text() + "".join(inspect.getsource(f) for f in funcs)
           + repr(consts)).encode()
    key = hashlib.sha1(src).hexdigest()[:12]
    if (station, key) in _MELT_CACHE:
        return _MELT_CACHE[(station, key)].copy()
    cache = mc.ROOT / "output" / "cache" / f"melt_coupled_{station}_{key}.pkl"
    if cache.exists():
        out = pd.read_pickle(cache)
    else:
        print(f"   {station}: gekoppelde Ts/G-smelt berekenen (eenmalig, wordt opgeslagen) ...", flush=True)
        out = melt_coupled(station)
        cache.parent.mkdir(parents=True, exist_ok=True)
        for old in cache.parent.glob(f"melt_coupled_{station}_*.pkl"):
            old.unlink()
        out.to_pickle(cache)
    _MELT_CACHE[(station, key)] = out
    return out.copy()


def ts_stats(h, col):
    d = h[[col, "ts_obs"]].dropna()
    diff = d[col] - d["ts_obs"]
    jja = d[d.index.month.isin([6, 7, 8])]
    dj = jja[col] - jja["ts_obs"]
    return {"n": len(d), "bias": diff.mean(), "rmse": np.sqrt((diff**2).mean()), "r": d[col].corr(d["ts_obs"]),
            "bias_jja": dj.mean(), "rmse_jja": np.sqrt((dj**2).mean()),
            "frac_smelt_mod_jja": (jja[col] >= -0.01).mean(),
            "frac_smelt_obs_jja": (jja["ts_obs"] >= -mc.T_MELT_TOL).mean()}


def plot(results, stats, out):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5), sharex=True, sharey=True)
    lim = (-50, 2)
    for ax, st in zip(axes.flat, mc.AWS_STATIONS):
        d = results[st].dropna()
        cmap = LinearSegmentedColormap.from_list(st, ["#ffffff", mc.COLORS[st]])
        hb = ax.hexbin(d["ts_obs"], d["ts_mod"], gridsize=70, extent=(*lim, *lim),
                       cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0)
        ax.plot(lim, lim, color="#52514e", lw=1, ls="--")
        s = stats.loc[st]
        ax.text(0.97, 0.05, f"bias = {s['bias']:+.2f} °C\nRMSE = {s['rmse']:.2f} °C\nn = {int(s['n']):,} h",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.set_title(st, loc="left", fontsize=11)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        mc.style(ax)
        ax.grid(axis="x", color="#e5e4e0", lw=0.8)
        fig.colorbar(hb, ax=ax, shrink=0.8, label="hours")
    for ax in axes[1]:
        ax.set_xlabel("Observed Ts from LWU (°C)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Modelled Ts (°C)")
    fig.suptitle("Modelled vs observed surface temperature (hourly)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(mc.ROOT / "output" / "surface_temperature"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results, rows = {}, []
    for st in mc.AWS_STATIONS:
        print(f"{st}: Ts en G oplossen ...", flush=True)
        h = modelled_ts(st)
        results[st] = h
        rows.append({"station": st, "G": "gekoppeld", **ts_stats(h, "ts_mod"),
                     "G_gem_JJA": h.loc[h.index.month.isin([6, 7, 8]), "g"].mean(),
                     "G_gem_DJF": h.loc[h.index.month.isin([12, 1, 2]), "g"].mean()})
        rows.append({"station": st, "G": "0", **ts_stats(h, "ts_mod_g0")})
        h.round(3).to_csv(out / f"ts_hourly_{st}.csv")
    allstats = pd.DataFrame(rows).set_index(["station", "G"])
    allstats.round(3).to_csv(out / "ts_stats.csv")
    print(allstats.round(2).to_string())
    print("\nPaper (fig. 2): bias -0.02 / +0.28 / +0.32 / +0.13 °C, RMSE 1.27 / 1.53 / 1.31 / 1.09 °C")
    stats = allstats.xs("gekoppeld", level="G")
    plot(results, stats, out / "ts_modelled_vs_observed.png")
    print(f"\nOutput opgeslagen in {out}")


if __name__ == "__main__":
    main()
