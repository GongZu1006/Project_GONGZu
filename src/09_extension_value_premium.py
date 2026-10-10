"""09_extension_value_premium.py - Extension, part A: where and when is the value premium in 2001-2025?

Run from the project root:  python3 src/09_extension_value_premium.py
Inputs : output/05_portfolio_returns_monthly.csv and output/05_factors_monthly.csv (aggregate monthly series),
         the public library file 6_Portfolios_2x3.csv under data/ (used as an independent check)
Outputs: output/09_*.csv, output/09_*.md, output/09_fig_*.png, output/09_extension_report.txt
Only aggregate series are read and written; no stock-level data.
"""
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
PORTS = ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]
LAGS = 6                                    # Newey-West (Bartlett) lags for monthly data
PAPER_MEAN, PAPER_T = 0.40, 2.91            # Fama and French (1993), Table 2, HML: % per month and t-statistic
ERAS = {"2001-07 to 2007-12": (200107, 200712),
        "2008-01 to 2015-12": (200801, 201512),
        "2016-01 to 2025-12": (201601, 202512)}
HML = "HML (value leg minus growth leg)"
SMALL = "Small-stock value spread (S/H - S/L)"
BIG = "Big-stock value spread (B/H - B/L)"
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
    df.to_csv(OUT / f"09_{name}.csv")
    (OUT / f"09_{name}.md").write_text(to_md(df, nd), encoding="utf-8")
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


def p_norm(z):
    return math.erfc(abs(z) / math.sqrt(2))           # two-sided normal p-value


def equal_means_p(stats):
    """Wald test that three independent sub-period means are equal (chi-square, 2 d.f.: p = exp(-W/2))."""
    assert len(stats) == 3
    w = np.array([1 / se ** 2 for _, se in stats])
    m = np.array([mm for mm, _ in stats])
    mbar = (w * m).sum() / w.sum()
    stat = float((w * (m - mbar) ** 2).sum())
    return stat, math.exp(-stat / 2)


def sub(s, a, b):
    return s[(s.index >= a) & (s.index <= b)]


def build(p, rf):
    value = 0.5 * (p["S/H"] + p["B/H"])
    growth = 0.5 * (p["S/L"] + p["B/L"])
    small = p["S/H"] - p["S/L"]
    big = p["B/H"] - p["B/L"]
    return {HML: value - growth,
            "Value leg, excess return": value - rf,
            "Growth leg, excess return": growth - rf,
            SMALL: small,
            BIG: big,
            "Small minus big value spread": small - big}


def find_library_file():
    bad = ("daily", "weekly", "wout", "ex_div")
    cands = [p for p in DATA.rglob("*.csv")
             if "6_portfolios" in p.name.lower().replace(" ", "_") and "2x3" in p.name.lower()
             and not any(k in p.name.lower() for k in bad)]
    return sorted(cands)[0] if cands else None


def library_vw(path):
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
            continue                                    # annual rows and column-header lines
        else:
            title = line
    for k, v in sections.items():
        if all(w in (k or "").lower() for w in ("value", "weighted", "monthly")):
            return pd.DataFrame(v, index=PORTS).T / 100.0
    raise RuntimeError("value-weighted monthly section not found in the library file")


def main():
    ports = pd.read_csv(OUT / "05_portfolio_returns_monthly.csv", index_col=0)[PORTS]
    fac = pd.read_csv(OUT / "05_factors_monthly.csv", index_col="ym")
    ports.index, fac.index = ports.index.astype(int), fac.index.astype(int)
    rf = fac["rf"].reindex(ports.index)

    lib_path = find_library_file()
    if lib_path is None:
        raise FileNotFoundError("6_Portfolios_2x3.csv not found under data/")
    lib = library_vw(lib_path).reindex(ports.index)
    log(f"months: {len(ports)} ({ports.index.min()} to {ports.index.max()}); "
        f"months missing in the library file: {int(lib.isna().any(axis=1).sum())}")

    mine, libs = build(ports, rf), build(lib, rf)
    log("check: HML rebuilt from my six portfolios vs HML saved in step 5, max abs difference (pp): "
        f"{(mine[HML] - fac['hml']).abs().max() * 100:.6f}")
    log("check: HML rebuilt from the library's six portfolios vs the library's HML factor, max abs difference (pp): "
        f"{(libs[HML] - fac['hml_b']).abs().max() * 100:.3f}")

    periods = {"Full sample": (int(ports.index.min()), int(ports.index.max())), **ERAS}

    # ---- Table 5.1: full sample ----
    rows = []
    for name in mine:
        m, se, t, n = nw_mean(mine[name])
        ml, sel, tl, _ = nw_mean(libs[name])
        rows.append([n, m * 100, t, m * 1200, ml * 100, tl])
    tab = pd.DataFrame(rows, index=list(mine),
                       columns=["Months", "Mean, this replication (% per month)", "NW t",
                                "Mean (% per year)", "Mean, library portfolios (% per month)", "NW t, library"])
    tab.index.name = "Series"
    tab["Months"] = tab["Months"].astype(int)
    save_table(tab, "table_full_sample", "Table 5.1. Value premium and its components, full sample (NW = Newey-West, 6 lags)")

    # ---- Table 5.2: is the modern premium different from the paper's 0.40% per month? ----
    se_paper = PAPER_MEAN / PAPER_T
    rows = []
    for label, s in [("HML, this replication", mine[HML]), ("HML, library portfolios", libs[HML])]:
        m, se, t, n = nw_mean(s)
        m, se = m * 100, se * 100
        z1 = (m - PAPER_MEAN) / se
        z2 = (m - PAPER_MEAN) / math.sqrt(se ** 2 + se_paper ** 2)
        rows.append([m, se, z1, p_norm(z1), z2, p_norm(z2)])
    tab = pd.DataFrame(rows, index=["HML, this replication", "HML, library portfolios"],
                       columns=["Mean (% per month)", "NW s.e. (pp)", "z vs 0.40 (this sample's s.e.)", "p (two-sided)",
                                "z vs 0.40 (adding the paper's s.e.)", "p (two-sided), combined"])
    tab.index.name = "Series"
    save_table(tab, "table_vs_paper", "Table 5.2. Test of H0: mean HML = 0.40% per month (the paper's estimate)", 4)

    # ---- Table 5.3 and 5.4: sub-periods ----
    rows, joint = [], []
    for name in mine:
        stm, stl = [], []
        for era, (a, b) in ERAS.items():
            m, se, t, n = nw_mean(sub(mine[name], a, b))
            ml, sel, tl, _ = nw_mean(sub(libs[name], a, b))
            stm.append((m, se))
            stl.append((ml, sel))
            rows.append([f"{name}; {era}", n, m * 100, t, ml * 100, tl])
        joint.append([name, *equal_means_p(stm), *equal_means_p(stl)])
    tab = pd.DataFrame(rows, columns=["Series; sub-period", "Months", "Mean, this replication (% per month)", "NW t",
                                      "Mean, library portfolios (% per month)", "NW t, library"]).set_index("Series; sub-period")
    tab["Months"] = tab["Months"].astype(int)
    save_table(tab, "table_subperiods", "Table 5.3. Sub-period means")
    tab = pd.DataFrame(joint, columns=["Series", "Wald stat, this replication", "p-value, this replication",
                                       "Wald stat, library", "p-value, library"]).set_index("Series")
    save_table(tab, "table_equality_tests", "Table 5.4. Wald tests that the three sub-period means are equal (2 d.f.)", 4)

    # ---- Table 5.5: HML variants ----
    variants = {"HML, this replication (security-level ME)": fac["hml"],
                "HML, this replication (firm-level ME)": fac["hml_firmME"],
                "HML, library factor": fac["hml_b"]}
    rows = []
    for s in variants.values():
        r = []
        for a, b in periods.values():
            m, se, t, n = nw_mean(sub(s, a, b))
            r += [m * 100, t]
        rows.append(r)
    cols = [f"{lab}: {k}" for lab in periods for k in ("mean (%)", "NW t")]
    tab = pd.DataFrame(rows, index=list(variants), columns=cols)
    tab.index.name = "Series"
    save_table(tab, "table_hml_variants", "Table 5.5. HML in three versions: full sample and sub-periods")

    # ---- Figures ----
    dates = pd.to_datetime(ports.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    w = 60
    rm = (mine[HML].rolling(w).mean() * 100).to_numpy()
    rl = (libs[HML].rolling(w).mean() * 100).to_numpy()
    band = (2 * mine[HML].rolling(w).std() * 100 / math.sqrt(w)).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, rm, label="This replication")
    ax.plot(dates, rl, "--", label="Library portfolios")
    ax.fill_between(dates, rm - band, rm + band, alpha=0.15, label="±2 s.e. band (i.i.d. approximation)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(PAPER_MEAN, color="gray", ls=":", label="Paper's 1963-1991 mean (0.40)")
    ax.set_title("HML: rolling 60-month mean (% per month)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "09_fig_rolling_hml.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (hi, lo, title) in zip(axes, [("S/H", "S/L", "Small stocks"), ("B/H", "B/L", "Big stocks")]):
        ax.plot(dates, (1 + ports[hi]).cumprod().to_numpy(), label=f"{hi} (value)")
        ax.plot(dates, (1 + ports[lo]).cumprod().to_numpy(), label=f"{lo} (growth)")
        ax.set_title(f"{title}: growth of $1")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "09_fig_value_vs_growth.png", dpi=150)
    plt.close(fig)

    names = [HML, SMALL, BIG]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.2
    for j, (lab, (a, b)) in enumerate(periods.items()):
        means, errs = [], []
        for nm in names:
            m, se, _, _ = nw_mean(sub(mine[nm], a, b))
            means.append(m * 100)
            errs.append(2 * se * 100)
        ax.bar(np.arange(len(names)) + (j - 1.5) * width, means, width, yerr=errs, capsize=3, label=lab)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(["HML", "Small-stock\nvalue spread", "Big-stock\nvalue spread"])
    ax.set_ylabel("Mean (% per month), error bars ±2 NW s.e.")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "09_fig_subperiod_means.png", dpi=150)
    plt.close(fig)

    (OUT / "09_extension_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nsaved the output/09_* files")


if __name__ == "__main__":
    main()
