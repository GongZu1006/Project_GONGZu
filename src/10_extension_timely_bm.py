"""10_extension_timely_bm.py - Extension, part B: does a timely BE/ME (June t market equity) revive HML?

Question: the baseline BE/ME divides book equity by market equity at December t-1 (as in Fama and French, 1993),
so by the June t formation date the denominator is six months stale. Here BE/ME is recomputed with June t
market equity (as proposed by Asness and Frazzini, 2013); everything else is unchanged (same eligible stocks,
same NYSE breakpoints rule, same size sort, same holding period, same value weights).

Run from the project root:  python3 src/10_extension_timely_bm.py
Inputs : processed/crsp_clean.parquet, processed/formation_universe.parquet, output/05_factors_monthly.csv
Outputs: output/10_*.csv, output/10_*.md, output/10_fig_*.png, output/10_timely_report.txt (aggregate numbers only)
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT = ROOT / "processed", ROOT / "output"
FIRST_YM, LAST_YM = 200107, 202512
PORTS = ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]
LAGS = 6
ERAS = {"2001-07 to 2007-12": (200107, 200712),
        "2008-01 to 2015-12": (200801, 201512),
        "2016-01 to 2025-12": (201601, 202512)}
LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def to_md(df, nd=3):
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for idx, row in zip(df.index, df.itertuples(index=False)):
        cells = [str(idx)] + [f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save_table(df, name, title, nd=3):
    df.to_csv(OUT / f"10_{name}.csv")
    (OUT / f"10_{name}.md").write_text(to_md(df, nd), encoding="utf-8")
    log(f"\n=== {title} ===")
    log(to_md(df, nd))


def nw_mean(x, lags=LAGS):
    """Mean, Newey-West (Bartlett) standard error of the mean, t-statistic, number of observations."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    m = x.mean()
    e = x - m
    s = e @ e / n
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(s / n)
    return m, se, m / se, n


def sub(s, a, b):
    return s[(s.index >= a) & (s.index <= b)]


def assign(el, bm_col):
    """NYSE breakpoints in June t (size median; BE/ME 30th and 70th percentiles), exactly as in step 5."""
    parts = []
    for t, d in el.groupby("t"):
        ny = d[d["nyse"]]
        size_bp = ny["me_jun"].median()
        lo, hi = ny[bm_col].quantile([0.3, 0.7])
        size = pd.Series(np.where(d["me_jun"] <= size_bp, "S", "B"), index=d.index)
        bm = pd.Series(np.where(d[bm_col] <= lo, "L", np.where(d[bm_col] <= hi, "M", "H")), index=d.index)
        a = d[["PERMNO", "t", "me_jun"]].copy()
        a["size"], a["bmg"], a["port"] = size, bm, size + "/" + bm
        parts.append(a)
    return pd.concat(parts)


def port_returns(r, a):
    rr = r.merge(a[["PERMNO", "t", "port"]], on=["PERMNO", "t"])
    g = rr.groupby(["ym", "port"])[["w", "wr"]].sum()
    pr = (g["wr"] / g["w"]).unstack("port")[PORTS]
    return pr


def hml_of(pr):
    return 0.5 * (pr["S/H"] + pr["B/H"]) - 0.5 * (pr["S/L"] + pr["B/L"])


def main():
    crsp = pd.read_parquet(PROC / "crsp_clean.parquet", columns=["PERMNO", "ym", "MthRet", "lag_me", "ret_ok"])
    uni = pd.read_parquet(PROC / "formation_universe.parquet",
                          columns=["PERMNO", "t", "nyse", "me_jun", "me_dec", "be", "bm_sec", "eligible"])
    fac = pd.read_csv(OUT / "05_factors_monthly.csv", index_col="ym")
    fac.index = fac.index.astype(int)

    el = uni[uni["eligible"]].copy()
    el["bm_base"] = el["be"] / el["me_dec"]
    el["bm_timely"] = el["be"] / el["me_jun"]
    log(f"eligible stock-years: {len(el):,}; formation years: {el['t'].nunique()}")
    log("check: BE/ME rebuilt as BE / ME_Dec vs bm_sec from step 4, max relative difference: "
        f"{((el['bm_base'] / el['bm_sec'] - 1).abs().max()):.2e}")

    # ---- return panel (same as step 5) ----
    r = crsp[crsp["ret_ok"] & crsp["ym"].between(FIRST_YM, LAST_YM)].copy()
    r["t"] = np.where(r["ym"] % 100 >= 7, r["ym"] // 100, r["ym"] // 100 - 1)
    r["w"] = r["lag_me"]
    r["wr"] = r["lag_me"] * r["MthRet"]

    a_base = assign(el, "bm_base")
    a_time = assign(el, "bm_timely")
    pr_base, pr_time = port_returns(r, a_base), port_returns(r, a_time)
    hml_b, hml_t = hml_of(pr_base), hml_of(pr_time)

    chk = (hml_b - fac["hml"].reindex(hml_b.index)).abs().max() * 100
    log(f"validation: baseline HML rebuilt here vs HML saved in step 5, max abs difference: {chk:.6f} pp "
        "(must be ~0, otherwise the pipeline differs from step 5)")
    log(f"months: {len(hml_b)}; months with a missing value (baseline / timely): "
        f"{int(hml_b.isna().sum())} / {int(hml_t.isna().sum())}")

    series = {"HML, baseline (BE/ME with December t-1 ME)": hml_b,
              "HML, timely (BE/ME with June t ME)": hml_t,
              "Difference (timely minus baseline)": hml_t - hml_b,
              "Small-stock value spread S/H-S/L, baseline": pr_base["S/H"] - pr_base["S/L"],
              "Small-stock value spread S/H-S/L, timely": pr_time["S/H"] - pr_time["S/L"],
              "Big-stock value spread B/H-B/L, baseline": pr_base["B/H"] - pr_base["B/L"],
              "Big-stock value spread B/H-B/L, timely": pr_time["B/H"] - pr_time["B/L"]}

    # ---- Table 5.6: full sample ----
    rows = []
    for name, s in series.items():
        m, se, t, n = nw_mean(s)
        sd = s.std()
        rows.append([n, m * 100, se * 100, t, sd * 100, m * 1200, m / sd * math.sqrt(12)])
    tab = pd.DataFrame(rows, index=list(series),
                       columns=["Months", "Mean (% per month)", "NW s.e. (pp)", "NW t", "Std (% per month)",
                                "Mean (% per year)", "Annualized mean/std"])
    tab.index.name = "Series"
    tab["Months"] = tab["Months"].astype(int)
    save_table(tab, "table_summary", "Table 5.6. Baseline and timely value factors, full sample")

    # ---- Table 5.7: sub-periods ----
    rows = []
    for name in list(series)[:3]:
        for era, (lo, hi) in {"Full sample": (FIRST_YM, LAST_YM), **ERAS}.items():
            m, se, t, n = nw_mean(sub(series[name], lo, hi))
            rows.append([f"{name}; {era}", n, m * 100, t])
    tab = pd.DataFrame(rows, columns=["Series; period", "Months", "Mean (% per month)", "NW t"]).set_index("Series; period")
    tab["Months"] = tab["Months"].astype(int)
    save_table(tab, "table_subperiods", "Table 5.7. Sub-period means")

    # ---- Table 5.8: correlations with the other factors ----
    other = fac.reindex(hml_b.index)
    cm = pd.DataFrame({"HML baseline": hml_b, "HML timely": hml_t,
                       "MKT-RF": other["mkt_rf"], "SMB": other["smb"]}).corr()
    cm.index.name = "Series"
    save_table(cm, "table_correlations", "Table 5.8. Correlations")

    # ---- Reclassification diagnostics ----
    m2 = a_base[["PERMNO", "t", "bmg", "me_jun"]].merge(a_time[["PERMNO", "t", "bmg"]], on=["PERMNO", "t"],
                                                          suffixes=("_base", "_time"))
    m2["moved"] = m2["bmg_base"] != m2["bmg_time"]
    by = m2.groupby("t").apply(lambda d: pd.Series({
        "stocks": len(d),
        "share_moved": d["moved"].mean(),
        "share_moved_value_weighted": (d["me_jun"] * d["moved"]).sum() / d["me_jun"].sum()}), include_groups=False)
    rk = el.groupby("t").apply(lambda d: d[["bm_base", "bm_timely"]].rank().corr().iloc[0, 1], include_groups=False)
    by["rank_corr_bm"] = rk
    by.index.name = "Formation year"
    by["stocks"] = by["stocks"].astype(int)
    save_table(by, "table_reclassification", "Table 5.9. Stocks changing BE/ME group when June ME replaces December ME")
    tm = pd.crosstab(m2["bmg_base"], m2["bmg_time"], normalize="index").reindex(index=["L", "M", "H"], columns=["L", "M", "H"])
    tm.index.name = "Baseline group (rows) / timely group (columns)"
    save_table(tm, "table_transition", "Table 5.10. Transition shares between baseline and timely BE/ME groups (pooled)")
    log(f"average share of stocks changing group: {by['share_moved'].mean():.1%}; value-weighted: "
        f"{by['share_moved_value_weighted'].mean():.1%}; average yearly rank correlation of BE/ME: {by['rank_corr_bm'].mean():.3f}")

    # ---- Composition under the timely sort ----
    comp = a_time.groupby(["t", "port"]).agg(n=("PERMNO", "size"), me=("me_jun", "sum")).reset_index()
    comp["share"] = comp["me"] / comp.groupby("t")["me"].transform("sum")
    avg = comp.groupby("port")[["n", "share"]].mean().loc[PORTS]
    avg.columns = ["Average number of stocks", "Average share of ME"]
    avg.index.name = "Portfolio (timely sort)"
    save_table(avg, "table_composition", "Table 5.11. Composition of the six portfolios under the timely sort")
    log(f"smallest number of stocks in any portfolio-year (timely): {int(comp['n'].min())}")

    # ---- Figures ----
    dates = pd.to_datetime(hml_b.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, (1 + hml_b).cumprod().to_numpy(), label="HML, baseline")
    ax.plot(dates, (1 + hml_t).cumprod().to_numpy(), "--", label="HML, timely")
    ax.set_title("Growth of $1: baseline and timely HML")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "10_fig_cumulative.png", dpi=150)
    plt.close(fig)

    diff = hml_t - hml_b
    w = 60
    rm = (diff.rolling(w).mean() * 100).to_numpy()
    band = (2 * diff.rolling(w).std() * 100 / math.sqrt(w)).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(dates, rm, label="Rolling 60-month mean of (timely - baseline)")
    ax.fill_between(dates, rm - band, rm + band, alpha=0.15, label="±2 s.e. band (i.i.d. approximation)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Difference between timely and baseline HML (% per month)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "10_fig_difference_rolling.png", dpi=150)
    plt.close(fig)

    eras = {"Full sample": (FIRST_YM, LAST_YM), **ERAS}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.25
    for j, (lab, s) in enumerate([("Baseline", hml_b), ("Timely", hml_t), ("Timely - baseline", diff)]):
        means, errs = [], []
        for lo, hi in eras.values():
            m, se, _, _ = nw_mean(sub(s, lo, hi))
            means.append(m * 100)
            errs.append(2 * se * 100)
        ax.bar(np.arange(len(eras)) + (j - 1) * width, means, width, yerr=errs, capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(eras)))
    ax.set_xticklabels(list(eras), fontsize=8)
    ax.set_ylabel("Mean (% per month), error bars ±2 NW s.e.")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "10_fig_subperiod_means.png", dpi=150)
    plt.close(fig)

    (OUT / "10_timely_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nsaved the output/10_* files")


if __name__ == "__main__":
    main()
