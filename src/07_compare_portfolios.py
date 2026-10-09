"""07_compare_portfolios.py - Compare my six portfolios with the French Data Library 6 Portfolios (2x3).

Run from the project root:  python3 src/07_compare_portfolios.py
Inputs : output/05_portfolio_returns_monthly.csv, output/05_portfolio_composition.csv,
         and the public library file 6_Portfolios_2x3.csv placed anywhere under data/
Outputs: output/07_*.csv and output/07_compare_portfolios_report.txt (aggregate numbers only)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
PORTS = ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]   # library order: SMALL Lo, ME1 BM2, SMALL Hi, BIG Lo, ME2 BM2, BIG Hi
LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def find_library_file():
    bad = ("daily", "weekly", "wout", "ex_div")
    cands = [p for p in DATA.rglob("*")
             if p.is_file() and p.name.lower().endswith(".csv")
             and "6_portfolios" in p.name.lower().replace(" ", "_") and "2x3" in p.name.lower()
             and not any(k in p.name.lower() for k in bad)]
    return sorted(cands)[0] if cands else None


def parse_library(path):
    """Split the file into titled sections of monthly rows (YYYYMM, six values)."""
    sections, title = {}, None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\d{6}\s*,", line):
            parts = [x.strip() for x in line.split(",")]
            vals = [float(x) if x not in ("", "-99.99", "-999") else np.nan for x in parts[1:7]]
            sections.setdefault(title, {})[int(parts[0])] = vals
        elif re.match(r"^\d{4}\s*,", line) or line.startswith(","):
            continue                       # annual rows and column-header lines
        else:
            title = line
    return sections


def pick(sections, *words):
    for k, v in sections.items():
        if all(w in (k or "").lower() for w in words):
            return pd.DataFrame(v, index=PORTS).T
    return None


def main():
    path = find_library_file()
    if path is None:
        raise FileNotFoundError("6_Portfolios_2x3.csv not found under data/")
    log(f"library file: {path.relative_to(ROOT)}")
    sections = parse_library(path)
    log("sections found: " + " | ".join(str(k) for k in sections))

    lib_vw = pick(sections, "value", "weighted", "monthly")
    if lib_vw is None:
        raise RuntimeError("value-weighted monthly section not found; send me the section titles above")
    lib_vw = lib_vw / 100.0
    nof = pick(sections, "number", "firms")
    size = pick(sections, "average", "firm", "size")

    ours = pd.read_csv(OUT / "05_portfolio_returns_monthly.csv", index_col=0)[PORTS]
    common = ours.index.intersection(lib_vw.index)
    log(f"months compared: {len(common)} ({common.min()} to {common.max()})")

    # ---- returns, portfolio by portfolio ----
    rows = {}
    for p in PORTS:
        d = pd.concat([ours.loc[common, p], lib_vw.loc[common, p]], axis=1, keys=["o", "l"]).dropna()
        diff = d["o"] - d["l"]
        rows[p] = [len(d), d["o"].mean() * 100, d["l"].mean() * 100, d["o"].corr(d["l"]),
                   diff.mean() * 100, diff.abs().mean() * 100, np.sqrt((diff ** 2).mean()) * 100]
    tab = pd.DataFrame(rows, index=["months", "mean ours (%)", "mean library (%)", "correlation",
                                    "mean diff (pp)", "mean abs diff (pp)", "RMSE (pp)"]).T.round(4)
    log("\n--- value-weighted returns of the six portfolios: this replication vs library ---")
    log(tab.to_string())
    tab.to_csv(OUT / "07_portfolio_agreement.csv")

    diff_df = ours.loc[common] - lib_vw.loc[common]
    by_year = (diff_df.abs().groupby(diff_df.index // 100).mean() * 100).round(3)
    log("\n--- mean absolute difference (pp) by calendar year and portfolio ---")
    log(by_year.to_string())
    by_year.to_csv(OUT / "07_portfolio_diff_by_year.csv")

    # ---- number of firms and average size (library values for July of year t vs my June-t formation) ----
    comp = pd.read_csv(OUT / "05_portfolio_composition.csv")
    n_ours = comp.pivot(index="t", columns="port", values="n")[PORTS]
    avg_ours = (comp.assign(avg=comp["me"] / comp["n"] / 1000.0)
                .pivot(index="t", columns="port", values="avg"))[PORTS]      # $ millions
    july = [t * 100 + 7 for t in n_ours.index]
    for label, lib, mine in [("number of firms", nof, n_ours), ("average firm size", size, avg_ours)]:
        if lib is None:
            log(f"\n(!) section for {label} not found in the library file")
            continue
        ref = lib.reindex(july)
        ref.index = mine.index
        ratio = (mine / ref).round(3)
        log(f"\n--- {label}: mine / library (July of each formation year) ---")
        log("average ratio by portfolio: " + ", ".join(f"{p}={ratio[p].mean():.3f}" for p in PORTS))
        log("minimum / maximum ratio:    " + ", ".join(f"{p}={ratio[p].min():.2f}/{ratio[p].max():.2f}" for p in PORTS))
        ratio.to_csv(OUT / f"07_ratio_{label.replace(' ', '_')}.csv")
        log(ratio.to_string())

    (OUT / "07_compare_portfolios_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nsaved the output/07_* files")


if __name__ == "__main__":
    main()