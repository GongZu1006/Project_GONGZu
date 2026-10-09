"""06_compare_benchmark.py - Tables and figures: my factors vs the French Data Library and the paper.

Run from the project root:  python3 src/06_compare_benchmark.py
Input : output/05_factors_monthly.csv (aggregate monthly series; no stock-level data)
Output: output/06_*.csv, output/06_*.md, output/06_fig_*.png, output/06_compare_report.txt
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
LINES = []
FACTORS = [("MKT-RF", "mkt_rf", "mkt_rf_b"), ("SMB", "smb", "smb_b"), ("HML", "hml", "hml_b")]
# Fama and French (1993), Table 2 (July 1963 - December 1991, 342 months): mean, std (% per month), t(mean)
PAPER = {"MKT-RF": (0.43, 4.54, 1.76), "SMB": (0.27, 2.89, 1.73), "HML": (0.40, 2.54, 2.91)}
# Correlations reported in the paper (Section 2.1.2 and Section 4.2)
PAPER_CORR = {("MKT-RF", "SMB"): 0.32, ("MKT-RF", "HML"): -0.38, ("SMB", "HML"): -0.08}


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
    df.to_csv(OUT / f"06_{name}.csv")
    md = to_md(df, nd)
    (OUT / f"06_{name}.md").write_text(md, encoding="utf-8")
    log(f"\n=== {title} ===")
    log(md)


def main():
    f = pd.read_csv(OUT / "05_factors_monthly.csv", index_col="ym")
    f.index = f.index.astype(int)
    f["date"] = pd.to_datetime(f.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    log(f"months: {len(f)} ({f.index.min()} to {f.index.max()})")

    # ---- Table A: summary statistics ----
    rows = {}
    for name, o, b in FACTORS:
        for label, col in [("this replication", o), ("French Data Library", b)]:
            s = f[col]
            rows[f"{name}: {label}"] = [len(s), s.mean() * 100, s.std() * 100,
                                        s.mean() / (s.std() / np.sqrt(len(s))), s.mean() * 1200]
        pm, ps, pt = PAPER[name]
        rows[f"{name}: original paper 1963-1991"] = [342, pm, ps, pt, pm * 12]
    tab_a = pd.DataFrame(rows, index=["Months", "Mean (% per month)", "Std (% per month)",
                                      "t(mean)", "Mean (% per year)"]).T
    tab_a.index.name = "Series"
    tab_a["Months"] = tab_a["Months"].astype(int)
    save_table(tab_a, "table_summary", "Table A. Summary statistics")

    # ---- Table B: correlations ----
    pairs = [("MKT-RF", "SMB", "mkt_rf", "smb", "mkt_rf_b", "smb_b"),
             ("MKT-RF", "HML", "mkt_rf", "hml", "mkt_rf_b", "hml_b"),
             ("SMB", "HML", "smb", "hml", "smb_b", "hml_b")]
    rows = {f"{a} and {b}": [f[c1].corr(f[c2]), f[d1].corr(f[d2]), PAPER_CORR[(a, b)]]
            for a, b, c1, c2, d1, d2 in pairs}
    tab_b = pd.DataFrame(rows, index=["This replication", "French Data Library",
                                      "Original paper 1963-1991"]).T
    tab_b.index.name = "Pair"
    save_table(tab_b, "table_correlations", "Table B. Correlations among the factors")

    # ---- Table C: agreement with the benchmark ----
    rows = {}
    diffs = {}
    for name, o, b in FACTORS:
        d = f[o] - f[b]
        diffs[name] = d
        slope, icpt = np.polyfit(f[b], f[o], 1)
        rows[name] = [f[o].corr(f[b]), slope, icpt * 100, d.mean() * 100,
                      d.mean() / (d.std() / np.sqrt(len(d))), d.abs().mean() * 100,
                      np.sqrt((d ** 2).mean()) * 100, d.abs().max() * 100]
    tab_c = pd.DataFrame(rows, index=["Correlation", "Slope", "Intercept (% per month)",
                                      "Mean difference (pp)", "t(mean difference)",
                                      "Mean abs difference (pp)", "RMSE (pp)", "Max abs difference (pp)"]).T
    tab_c.index.name = "Factor"
    save_table(tab_c, "table_agreement", "Table C. Agreement with the French Data Library (this replication minus library)")
    diffs = pd.DataFrame(diffs)

    # ---- Table D: where do the differences occur? ----
    by_month = diffs.abs().groupby(f.index % 100).mean() * 100
    by_month.index.name = "Calendar month"
    save_table(by_month, "diff_by_calendar_month", "Table D1. Mean absolute difference (pp) by calendar month", 3)
    by_year = diffs.abs().groupby(f.index // 100).mean() * 100
    by_year.index.name = "Year"
    save_table(by_year, "diff_by_year", "Table D2. Mean absolute difference (pp) by year", 3)
    is_july = (f.index % 100) == 7
    log("\nmean absolute difference (pp): July vs other months")
    log(pd.DataFrame({"July": diffs.abs()[is_july].mean() * 100,
                      "Other months": diffs.abs()[~is_july].mean() * 100}).round(3).to_string())
    top = []
    for name, o, b in FACTORS:
        for ym in diffs[name].abs().sort_values(ascending=False).index[:10]:
            top.append([name, ym, f.loc[ym, o] * 100, f.loc[ym, b] * 100, diffs.loc[ym, name] * 100])
    top = pd.DataFrame(top, columns=["Factor", "ym", "This replication (%)", "Library (%)", "Difference (pp)"])
    top.index = range(1, len(top) + 1)
    top.index.name = "Rank"
    save_table(top, "diff_top_months", "Table D3. Ten largest absolute differences per factor", 3)

    # ---- Figures ----
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, (name, o, b) in zip(axes, FACTORS):
        ax.plot(f["date"], (1 + f[o]).cumprod(), "-", label="This replication")
        ax.plot(f["date"], (1 + f[b]).cumprod(), "--", label="French Data Library")
        ax.set_title(f"{name}: growth of $1")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "06_fig_cumulative_returns.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (name, o, b) in zip(axes, FACTORS):
        lim = [min(f[b].min(), f[o].min()) * 100, max(f[b].max(), f[o].max()) * 100]
        ax.scatter(f[b] * 100, f[o] * 100, s=10, alpha=0.6)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("French Data Library (% per month)")
        ax.set_ylabel("This replication (% per month)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "06_fig_scatter.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, (name, o, b) in zip(axes, FACTORS):
        ax.bar(f["date"], diffs[name] * 100, width=25)
        ax.set_title(f"{name}: this replication minus French Data Library (pp per month)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "06_fig_differences.png", dpi=150)
    plt.close(fig)

    (OUT / "06_compare_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nsaved the output/06_* files")


if __name__ == "__main__":
    main()