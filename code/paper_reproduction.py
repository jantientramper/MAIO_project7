"""Additional analyses/figures for Kuipers Munneke et al. (2018), K-transect SEB.

Uses the existing melt_comparison.py module. Ground heat flux is deliberately
set to zero, as requested; therefore these are reproductions using a simplified
SEB, not a bit-for-bit reproduction of the published subsurface model.

Run from the same project layout as melt_comparison.py:
    python paper_reproduction.py --out output/paper_reproduction
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import melt_comparison as mc
from matplotlib.lines import Line2D


STATIONS = mc.AWS_STATIONS
COMPONENTS = ["SWnet", "LWnet", "SHF", "LHF"]
COLORS = mc.COLORS
MIN_COVERAGE = 0.80
PAPER_START_YEAR = 2003
PAPER_END_YEAR = 2016


def _hours_and_coverage(ts: pd.DataFrame) -> pd.DataFrame:
    out = ts.copy().sort_index()
    dt = out.index.to_series().shift(-1).sub(out.index.to_series()).dt.total_seconds() / 3600
    positive = dt[dt > 0]
    fallback = float(positive[positive <= 1].median()) if (positive <= 1).any() else 1.0
    out["_dt_h"] = dt.fillna(fallback).clip(lower=0, upper=1)
    return out


def calculate_period_summary(station: str, freq: str = "M", start=None, end=None,
                             min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    ts = _hours_and_coverage(mc.melt_station(station))
    if start is not None:
        ts = ts.loc[pd.Timestamp(start):]
    if end is not None:
        ts = ts.loc[ts.index < pd.Timestamp(end)]
    if ts.empty:
        return pd.DataFrame()
    periods = ts.index.to_period(freq)
    # duration represented by each timestamp, in seconds
    ts["_melt_mj_m2"] = ts["M"] * ts["_dt_h"] * 3600 / 1e6
    ts["_melt_mmwe"] = ts["M"] * ts["_dt_h"] * 3600 / mc.LF
    ts["_valid"] = ts["M"].notna()
    expected = pd.Series(
        ((periods.end_time - periods.start_time).total_seconds() + 1e-9) / 3600,
        index=ts.index,
    ).groupby(periods).first()
    g = ts.groupby(periods)
    coverage = (ts["_dt_h"] * ts["_valid"]).groupby(periods).sum() / expected
    result = g[COMPONENTS + ["M"]].mean()
    result["Rnet"] = result["SWnet"] + result["LWnet"]
    result["melt_energy_MJ_m2"] = ts["_melt_mj_m2"].groupby(periods).sum(min_count=1)
    result["melt_mmwe"] = ts["_melt_mmwe"].groupby(periods).sum(min_count=1)
    result["melt_hours"] = (ts["_dt_h"] * (ts["M"] > 0)).groupby(periods).sum()
    result["coverage"] = coverage.clip(upper=1)
    result.loc[result["coverage"] < min_coverage, result.columns.difference(["coverage"])] = np.nan
    result.index.name = "period"
    return result


def calculate_jja_annual(station: str, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    ts = _hours_and_coverage(mc.melt_station(station))
    ts = ts[ts.index.month.isin([6, 7, 8])].copy()
    if ts.empty:
        return pd.DataFrame()
    ts["year"] = ts.index.year
    ts["_melt_mmwe"] = ts["M"] * ts["_dt_h"] * 3600 / mc.LF
    ts["_melt_MJ_m2"] = ts["M"] * ts["_dt_h"] * 3600 / 1e6
    ts["_valid"] = ts["M"].notna()
    # Use 92 days (2208 h) for JJA; leap years do not change JJA length.
    g = ts.groupby("year")
    coverage = (ts["_dt_h"] * ts["_valid"]).groupby(ts["year"]).sum() / (92 * 24)
    out = g[COMPONENTS + ["M"]].mean()
    out["Rnet"] = out["SWnet"] + out["LWnet"]
    out["melt_mmwe"] = g["_melt_mmwe"].sum(min_count=1)
    out["melt_energy_MJ_m2"] = g["_melt_MJ_m2"].sum(min_count=1)
    out["melt_hours"] = (ts["_dt_h"] * (ts["M"] > 0)).groupby(ts["year"]).sum()
    out["coverage"] = coverage.clip(upper=1)
    out.loc[out["coverage"] < min_coverage, out.columns.difference(["coverage"])] = np.nan
    out.index.name = "year"
    return out


def create_coverage_table(stations=STATIONS) -> pd.DataFrame:
    rows = []
    for st in stations:
        ts = _hours_and_coverage(mc.melt_station(st))
        if ts.empty:
            continue
        ts["year"] = ts.index.year
        for year, d in ts.groupby("year"):
            expected = 365 + int(pd.Timestamp(year, 12, 31).is_leap_year)
            expected_hours = expected * 24
            valid = d["M"].notna()
            hours = float((d["_dt_h"] * valid).sum())
            jja = d[d.index.month.isin([6, 7, 8])]
            jja_hours = float((jja["_dt_h"] * jja["M"].notna()).sum())
            rows.append({"station": st, "year": int(year), "n_timestamps": len(d),
                         "valid_SEB_timestamps": int(valid.sum()),
                         "annual_coverage_fraction": min(hours / expected_hours, 1),
                         "JJA_valid_timestamps": int(jja["M"].notna().sum()),
                         "JJA_coverage_fraction": min(jja_hours / (92 * 24), 1),
                         "annual_usable_80pct": hours / expected_hours >= MIN_COVERAGE,
                         "JJA_usable_80pct": jja_hours / (92 * 24) >= MIN_COVERAGE})
    return pd.DataFrame(rows).sort_values(["station", "year"])


def create_monthly_climatology(stations=STATIONS) -> pd.DataFrame:
    rows = []
    for st in stations:
        monthly = calculate_period_summary(st, "M")
        if monthly.empty:
            continue
        monthly["month"] = monthly.index.month
        for month, d in monthly.groupby("month"):
            row = {"station": st, "month": int(month)}
            for c in COMPONENTS + ["M", "Rnet"]:
                row[c] = d[c].mean()
            row["n_months"] = int(d["M"].notna().sum())
            rows.append(row)
    return pd.DataFrame(rows)


def create_jja_correlations(stations=STATIONS, start_year=PAPER_START_YEAR,
                            end_year=PAPER_END_YEAR) -> pd.DataFrame:
    rows = []
    predictors = ["SWnet", "LWnet", "Rnet", "SHF", "LHF"]
    for st in stations:
        d = calculate_jja_annual(st)
        d = d.loc[(d.index >= start_year) & (d.index <= end_year)]
        d = d.dropna(subset=["M"])
        for var in predictors:
            pair = d[["M", var]].dropna()
            if len(pair) >= 3 and pair["M"].nunique() > 1 and pair[var].nunique() > 1:
                r, p = stats.pearsonr(pair["M"], pair[var])
            else:
                r, p = np.nan, np.nan
            rows.append({"station": st, "component": var, "pearson_r": r,
                         "p_value": p, "n_years": len(pair)})
    return pd.DataFrame(rows)


def _hydrological_year_summary(station: str, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    ts = _hours_and_coverage(mc.melt_station(station))
    if ts.empty:
        return pd.DataFrame()
    # Label a hydrological year by the year in which it ends (e.g. Sep 2003-Aug 2004 -> 2004).
    ts["hydro_year"] = ts.index.year + (ts.index.month >= 9).astype(int)
    ts["_melt_mmwe"] = ts["M"] * ts["_dt_h"] * 3600 / mc.LF
    ts["_melt_MJ_m2"] = ts["M"] * ts["_dt_h"] * 3600 / 1e6
    ts["_valid"] = ts["M"].notna()
    g = ts.groupby("hydro_year")
    expected = pd.Series(365 * 24, index=ts.index)
    # Use 366 days for hydrological years containing Feb 29.
    leap = (ts.index.month == 2) & (ts.index.day == 29)
    years = ts["hydro_year"]
    expected_by_year = {int(y): (366 if leap[years == y].any() else 365) * 24 for y in years.unique()}
    coverage = (ts["_dt_h"] * ts["_valid"]).groupby(years).sum() / pd.Series(expected_by_year)
    out = g[COMPONENTS + ["M"]].mean()
    out["Rnet"] = out["SWnet"] + out["LWnet"]
    out["melt_mmwe"] = g["_melt_mmwe"].sum(min_count=1)
    out["melt_energy_MJ_m2"] = g["_melt_MJ_m2"].sum(min_count=1)
    out["coverage"] = coverage.clip(upper=1)
    out.loc[out["coverage"] < min_coverage, out.columns.difference(["coverage"])] = np.nan
    out.index.name = "year"
    return out


def _trend_row(station, period_name, variable, series, min_years=5):
    series = series.dropna()
    n = len(series)
    if n < min_years:
        return None
    x = series.index.to_numpy(dtype=float)
    fit = stats.linregress(x, series.to_numpy(dtype=float))
    return {"station": station, "period": period_name, "variable": variable,
            "trend_per_year": fit.slope, "trend_per_decade": fit.slope * 10,
            "intercept": fit.intercept, "r": fit.rvalue, "p_value": fit.pvalue,
            "n_years": n, "first_year": int(x.min()), "last_year": int(x.max())}


def create_trend_table(stations=STATIONS, min_years=5,
                       start_year=PAPER_START_YEAR, end_year=PAPER_END_YEAR) -> pd.DataFrame:
    variables = COMPONENTS + ["Rnet", "M", "melt_mmwe", "melt_energy_MJ_m2"]
    rows = []
    for st in stations:
        for period_name, d in [("JJA", calculate_jja_annual(st)),
                               ("Hydrological year (Sep-Aug)", _hydrological_year_summary(st))]:
            if not d.empty:
                d = d.loc[(d.index >= start_year) & (d.index <= end_year)]
            for var in variables:
                if var in d:
                    row = _trend_row(st, period_name, var, d[var], min_years)
                    if row:
                        rows.append(row)
    return pd.DataFrame(rows)

def create_validation_table(stations=("AWS5", "AWS6"), freq="Y",
                            start_year=PAPER_START_YEAR, end_year=PAPER_END_YEAR) -> pd.DataFrame:
    rows = []
    for st in stations:
        d = calculate_period_summary(st, freq)
        raw = mc.aggregate(mc.melt_station(st), freq)
        both = d[["melt_mmwe"]].join(raw[["obs_melt_mmwe_ice"]], how="inner").dropna()
        if not both.empty:
            year_index = both.index.year if isinstance(both.index, pd.DatetimeIndex) else pd.Index(both.index).astype(str).str[:4].astype(int)
            both = both.loc[(year_index >= start_year) & (year_index <= end_year)]
        if both.empty:
            rows.append({"station": st, "n_periods": 0, "bias_mmwe": np.nan,
                         "MAE_mmwe": np.nan, "RMSE_mmwe": np.nan, "correlation": np.nan})
            continue
        err = both["melt_mmwe"] - both["obs_melt_mmwe_ice"]
        corr = both["melt_mmwe"].corr(both["obs_melt_mmwe_ice"]) if len(both) > 1 else np.nan
        rows.append({"station": st, "n_periods": len(both), "bias_mmwe": err.mean(),
                     "MAE_mmwe": err.abs().mean(), "RMSE_mmwe": np.sqrt((err**2).mean()),
                     "correlation": corr})
    return pd.DataFrame(rows)


def plot_monthly_seb(stations=STATIONS, out="figure4_monthly_seb.png"):
    fluxes = ["SWnet", "LWnet", "SHF", "LHF", "M"]
    fig, axes = plt.subplots(len(stations), 1, figsize=(13, 3.0 * len(stations)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, stations):
        monthly = calculate_period_summary(st, "M")
        if monthly.empty:
            continue
        x = monthly.index.to_timestamp(how="start")
        for c in fluxes:
            ax.plot(x, monthly[c], lw=1.15, label=c)
        ax.axhline(0, color="black", lw=.7)
        ax.set_ylabel(f"{st}\nW m$^{{-2}}")
        ax.grid(True, alpha=.25)
    axes[0].legend(ncol=5, fontsize=8, frameon=False, loc="best")
    axes[-1].set_xlabel("Year")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(axis="x", labelrotation=45)
    fig.suptitle("Monthly mean surface energy-balance fluxes (G = 0)")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_hourly_case_study(station="AWS9", start="2012-07-07", end="2012-07-14",
                           out="figure5_hourly_case_study.png"):
    ts = mc.melt_station(station)
    ts = ts.loc[(ts.index >= pd.Timestamp(start)) & (ts.index < pd.Timestamp(end))]
    if ts.empty:
        warnings.warn(f"No observations for {station} in {start} to {end}; case-study plot skipped.")
        return
    df = mc.load_aws(station) if station in mc.AWS_STATIONS else None

    ccol = {"SWnet": "#12A1CD", "LWnet": "#DEAF14", "SHF": "#34C97A", "LHF": "#BB69DE"}

    fig, ax = plt.subplots(figsize=(9, 5))

    line_colors = []
    for c in COMPONENTS + ["M"]:
        if c in ts:
            color = ccol.get(c, "0.3")          # melt (M) falls back to gray 0.3
            label = "Melt" if c == "M" else c
            ax.plot(ts.index, ts[c], color=color, lw=2.5, label=label)
            line_colors.append(color)

    # left axis: -100..300 ticks, extra headroom above for the wind curve
    ax.set_ylim(-100, 504)
    ax.set_yticks([-100, 0, 100, 200, 300])
    ax.set_ylabel("Energy flux (W m$^{-2}$)", fontsize=13, y=0.33)
    ax.set_xlim(ts.index[0], ts.index[-1])

    # x axis: one label per day centred at noon, dotted line at each midnight
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    for d in days:
        ax.axvline(d, color="k", ls=":", lw=0.8)
    noons = [d + pd.Timedelta(hours=12) for d in days[:-1]]
    ax.set_xticks(noons)
    ax.set_xticklabels([d.strftime("%-d-%b-%Y") for d in noons], fontsize=11)
    ax.tick_params(axis="x", length=0)
    ax.grid(True, axis="y", ls=":", color="k", lw=0.8)

    # wind speed on the right axis; 0 m/s sits at 300 W m-2, 7.5 m/s ≈ 100 W m-2 higher
    ax2 = ax.twinx()
    if df is not None and "ff" in df:
        wind = df["ff"].reindex(ts.index)
        ax2.plot(wind.index, wind, color="#5E5D5DB1", lw=2.5)
    ax2.set_ylim(-30, 15)                    # maps onto ax ylim (-100, 504)
    ax2.set_yticks([0, 7.5, 15])
    ax2.set_yticklabels(["0", "7.5", "15"])
    ax2.set_ylabel("10-m wind speed (m s$^{-1}$)", fontsize=13, rotation=270, labelpad=20, y=0.75)

    # legend: upper right, colored line handles, black text, with box
    ax.legend(ncol=2, frameon=True, fontsize=12, loc="upper right",
              handlelength=2.5, columnspacing=1)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

# def plot_hourly_case_study(station="AWS9", start="2012-07-07", end="2012-07-14",
#                            out="figure5_hourly_case_study.png"):
#     ts = mc.melt_station(station)
#     ts = ts.loc[(ts.index >= pd.Timestamp(start)) & (ts.index < pd.Timestamp(end))]
#     if ts.empty:
#         warnings.warn(f"No observations for {station} in {start} to {end}; case-study plot skipped.")
#         return
#     df = mc.load_aws(station) if station in mc.AWS_STATIONS else None
#     fig, ax = plt.subplots(figsize=(11, 5))
#     for c in COMPONENTS + ["M"]:
#         if c in ts:
#             ax.plot(ts.index, ts[c], lw=1.2, label=c)
#     ax.axhline(0, color="black", lw=.7)
#     ax.set_ylabel("Energy flux (W m$^{-2}$)")
#     ax.set_xlabel("Date")
#     if df is not None and "ff" in df:
#         wind = df["ff"].reindex(ts.index)
#         ax2 = ax.twinx()
#         ax2.plot(wind.index, wind, color="0.25", lw=1, ls="--", label="Wind speed")
#         ax2.set_ylabel("Wind speed (m s$^{-1}$)")
#         h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
#         ax.legend(h1 + h2, l1 + l2, ncol=3, frameon=False, fontsize=8)
#     else:
#         ax.legend(ncol=3, frameon=False, fontsize=8)
#     ax.set_title(f"{station}: {start} to {end}")
#     ax.grid(True, alpha=.25)
#     fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_jja_fluxes_by_melt_rank(stations=STATIONS, out="figure6_jja_fluxes_by_melt_rank.png"):
    fluxes = ["SWnet", "LWnet", "SHF", "LHF"]
    fig, axes = plt.subplots(len(stations), 1, figsize=(12, 3.4 * len(stations)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, st in zip(axes, stations):
        d = calculate_jja_annual(st).dropna(subset=["M"]).sort_values("M").reset_index()
        if d.empty:
            continue
        x = np.arange(len(d)); width = 0.8 / len(fluxes)
        for i, c in enumerate(fluxes):
            ax.bar(x + (i - (len(fluxes)-1)/2) * width, d[c], width=width, label=c)
        ax.axhline(0, color="black", lw=.7)
        ax.set_ylabel(f"{st}\nW m$^{{-2}}")
        ax.set_xticks(x, d["year"].astype(str), rotation=45)
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(ncol=len(fluxes), fontsize=8, frameon=False)
    axes[-1].set_xlabel("Year (ordered by increasing JJA mean melt flux)")
    fig.suptitle("JJA energy-balance fluxes by year, ranked by melt (G = 0)")
    fig.tight_layout(); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)

def plot_annual_jja_melt(stations=STATIONS, out="figure8_annual_jja_melt.png", title=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    all_years = []
    for st in stations:
        annual = calculate_period_summary(st, "Y")
        jja = calculate_jja_annual(st)
        color = COLORS.get(st) if isinstance(COLORS, dict) else None
        if not annual.empty:
            years = annual.index.year
            all_years.extend(list(years))
            line, = ax.plot(years, annual["melt_energy_MJ_m2"], marker="o", ms=8,
                            lw=2.5, ls="-", color=color, label=f"{st} annual")
            color = line.get_color()
        if not jja.empty:
            all_years.extend(list(jja.index))
            ax.plot(jja.index, jja["melt_energy_MJ_m2"], marker="o", ms=8,
                    markerfacecolor="white", markeredgecolor=color, lw=2.5,
                    ls="--", color=color, label=f"{st} JJA")
    if all_years:
        first_year, last_year = int(min(all_years)), int(max(all_years))
        ax.set_xticks(np.arange(first_year, last_year + 1, 1))
        ax.set_xlim(first_year + 0.75, last_year + 0.25)
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("Total melt energy (MJ m$^{-2}$)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    if title:
        ax.set_title(title, fontsize=11, loc="left", pad=40)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)

    style_handles = [
        Line2D([0], [0], color="black", lw=2.5, ls="-",
            marker="o", markersize=7, label="Annual"),
        Line2D([0], [0], color="black", lw=2.5, ls="--",
            marker="o", markerfacecolor="white",
            markeredgecolor="black", markersize=7, label="JJA"),]

    leg1 = ax.legend(
        handles=style_handles,
        loc="lower left", bbox_to_anchor=(0, 1.01),
        frameon=False)

    ax.add_artist(leg1)

    station_handles = [Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            color=COLORS[st],
            markersize=8,
            label=st)
        for st in stations]

    leg2 = ax.legend(
        handles=station_handles,
        loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2,
        frameon=False,
        labelcolor="black",
        handletextpad=0.6)

    ax.add_artist(leg2)

    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)


def table3_paper_layout(correlations: pd.DataFrame) -> pd.DataFrame:
    component_order = ["SWnet", "LWnet", "Rnet", "SHF", "LHF"]
    station_order = [s for s in STATIONS if s in set(correlations["station"])]
    labels = {"SWnet": "SWnet", "LWnet": "LWnet",
              "Rnet": "Rnet", "SHF": "H", "LHF": "L"}
    wide = correlations.pivot(index="component", columns="station", values="pearson_r")
    wide = wide.reindex(index=component_order, columns=station_order)
    wide.index = [labels.get(v, v) for v in wide.index]
    wide.columns = [s.replace("AWS", "S") for s in wide.columns]
    wide.columns.name = "Correlation coefficient with M"
    return wide


def table4_paper_layout(trends: pd.DataFrame) -> pd.DataFrame:
    paper_stations = [s for s in ["AWS5", "AWS6", "AWS9"] if s in set(trends["station"])]
    d = trends[trends["station"].isin(paper_stations)].copy()
    d["period_short"] = d["period"].map({"Hydrological year (Sep-Aug)": "Hydrological year",
                                          "JJA": "JJA"})
    d["column"] = d["station"].str.replace("AWS", "S", regex=False) + " " + d["period_short"]
  
    flux_order = ["SWnet", "LWnet", "SHF", "LHF", "M"]
    wide = d.pivot(index="variable", columns="column", values="trend_per_year")
    col_order = [f"{st.replace('AWS', 'S')} {period}" for st in paper_stations
                 for period in ["Hydrological year", "JJA"]]
    wide = wide.reindex(index=flux_order, columns=col_order)
    wide.index = [{"SWnet": "SWnet", "LWnet": "LWnet",
                   "SHF": "H", "LHF": "L", "M": "M"}.get(v, v)
                  for v in wide.index]
    wide.index.name = "Flux trend (W m$^{-2}$ y$^{-1}$)"
    return wide


def print_paper_tables(correlations: pd.DataFrame, trends: pd.DataFrame) -> None:
    t3 = table3_paper_layout(correlations)
    print(f"\nTABLE 3. Correlation coefficients (R) between June, July, August (JJA) mean melt flux and other SEB components ({PAPER_START_YEAR}–{PAPER_END_YEAR})")
    print(t3.to_string(float_format=lambda x: f"{x:.2f}"))

    t4 = table4_paper_layout(trends)
    sig_lookup = {(r.station, r.period, r.variable): bool(r.p_value < 0.10)
                  for r in trends.itertuples() if pd.notna(r.p_value)}
    raw_variables = {"SWnet": "SWnet", "LWnet": "LWnet",
                     "H": "SHF", "L": "LHF", "M": "M"}
    def fmt_trend(value, variable, col):
        if pd.isna(value):
            return "—"
        station_label, period_label = col.split(" ", 1)
        station = station_label.replace("S", "AWS", 1)
        period = "JJA" if period_label == "JJA" else "Hydrological year (Sep-Aug)"
        significant = sig_lookup.get((station, period, raw_variables.get(variable, variable)), False)
        rendered = f"{value:.2f}"
        # ANSI bold renders as bold in many terminals; significance is also saved in CSV.
        return f"\033[1m{rendered}\033[0m" if significant else rendered
    display = t4.copy().astype(object)
    for variable in display.index:
        for col in display.columns:
            display.loc[variable, col] = fmt_trend(t4.loc[variable, col], variable, col)
    print(f"\nTABLE 4. Trends in SEB components ({PAPER_START_YEAR}–{PAPER_END_YEAR}; W m^-2 y^-1); bold indicates p < 0.10")
    print(display.to_string())

FIG8_METHODS = {
    "seb_ts": "Modelled Ts with subsurface heat flux G (coupled)",
    "threshold": "Ts from observed LWU, melt threshold 0.5 K, G = 0 (original)",
}


def plot_figure8_both_methods(out):
    """Figuur 8 voor beide smeltmethoden, elk in een eigen bestand."""
    original = mc.MELT_METHOD
    try:
        for method, title in FIG8_METHODS.items():
            mc.MELT_METHOD = method
            # zelfde y-as voor beide versies, zodat ze direct te vergelijken zijn
            plot_annual_jja_melt(out=Path(out) / f"figure8_annual_jja_melt_{method}.png", title=title,
                                 ylim=(0, 2400))
    finally:
        mc.MELT_METHOD = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(mc.ROOT / "output" / "paper_reproduction"))
    parser.add_argument("--case-start", default="2012-08-15")
    parser.add_argument("--case-end", default="2012-08-21")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    coverage = create_coverage_table()
    coverage.to_csv(out / "table_data_coverage.csv", index=False)
    monthly = create_monthly_climatology()
    monthly.to_csv(out / "table_monthly_seb_climatology.csv", index=False)
    corr = create_jja_correlations(start_year=PAPER_START_YEAR, end_year=PAPER_END_YEAR)
    corr.to_csv(out / "table3_jja_correlations.csv", index=False)
    corr.to_csv(out / "table_jja_correlations.csv", index=False)
    trends = create_trend_table(start_year=PAPER_START_YEAR, end_year=PAPER_END_YEAR)
    trends.to_csv(out / "table4_seb_trends.csv", index=False)
    trends.to_csv(out / "table_jja_trends.csv", index=False)
    validation = create_validation_table(start_year=PAPER_START_YEAR, end_year=PAPER_END_YEAR)
    validation.to_csv(out / "table_melt_validation.csv", index=False)

    # Print the requested paper tables to the terminal as well as saving CSVs.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_rows", 200)
    print(f"Tables restricted to year labels {PAPER_START_YEAR}–{PAPER_END_YEAR} (inclusive).")
    print_paper_tables(corr, trends)
    table3_paper_layout(corr).to_csv(out / "table3_paper_layout.csv")
    table4_paper_layout(trends).to_csv(out / "table4_paper_layout.csv")

    for st in STATIONS:
        annual = calculate_jja_annual(st)
        annual.insert(0, "station", st)
        annual.to_csv(out / f"jja_annual_{st}.csv")

    plot_monthly_seb(out=out / "figure4_monthly_seb.png")
    plot_hourly_case_study(start=args.case_start, end=args.case_end,
                           out=out / "figure5_hourly_case_study.png")
    plot_jja_fluxes_by_melt_rank(out=out / "figure6_jja_fluxes_by_melt_rank.png")
    plot_annual_jja_melt(out=out / "figure8_annual_jja_melt.png")
    plot_figure8_both_methods(out)
    print(f"Saved reproduction outputs to {out}")
    print("Note: G is fixed at zero. Figure/table numbers are intended as working analogues; check definitions against the paper.")

if __name__ == "__main__":
    main()
