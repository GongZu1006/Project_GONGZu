"""05_portfolios_factors.py - 2x3 portfolios, SMB, HML and MKT-RF (July 2001 - December 2025).

Run from the project root:  python3 src/05_portfolios_factors.py
Inputs : processed/crsp_clean.parquet, processed/formation_universe.parquet,
         "F-F factors and RF.csv" under data/
Outputs (aggregate series and counts only): output/05_*.csv, output/05_factors_report.txt
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, PROC, OUT = ROOT / "data", ROOT / "processed", ROOT / "output"
FIRST_YM, LAST_YM = 200107, 202512
PORTS = ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]
LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def load_ff():
    path = next(iter(sorted(DATA.rglob("F-F factors and RF.csv"))), None)
    if path is None:
        raise FileNotFoundError("F-F factors and RF.csv not found under data/")
    pat = re.compile(r"^\s*(\d{6})\s*,")
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pat.match(line):
            parts = [x.strip() for x in line.split(",")]
            rows.append([int(parts[0])] + [float(x) for x in parts[1:5]])
    ff = pd.DataFrame(rows, columns=["ym", "mkt_rf_b", "smb_b", "hml_b", "rf"]).set_index("ym")
    return ff / 100.0                      # percent -> decimal


def assign(uni, bm_col):
    """NYSE breakpoints in June t; returns (assignment, breakpoints)."""
    el = uni[uni["eligible"]]
    parts, bps = [], []
    for t, d in el.groupby("t"):
        ny = d[d["nyse"]]
        size_bp = ny["me_jun"].median()
        lo, hi = ny[bm_col].quantile([0.3, 0.7])
        size = pd.Series(np.where(d["me_jun"] <= size_bp, "S", "B"), index=d.index)
        bm = pd.Series(np.where(d[bm_col] <= lo, "L", np.where(d[bm_col] <= hi, "M", "H")), index=d.index)
        a = d[["PERMNO", "t", "me_jun"]].copy()
        a["port"] = size + "/" + bm
        parts.append(a)
        bps.append({"t": t, "n_nyse": len(ny), "size_median_ME": size_bp, "bm_p30": lo, "bm_p70": hi})
    return pd.concat(parts), pd.DataFrame(bps)


def port_returns(r, a):
    rr = r.merge(a[["PERMNO", "t", "port"]], on=["PERMNO", "t"])
    g = rr.groupby(["ym", "port"])[["w", "wr"]].sum()
    pr = (g["wr"] / g["w"]).unstack("port")[PORTS]
    cnt = rr.groupby(["ym", "port"]).size().unstack("port")[PORTS]
    return pr, cnt


def factors(pr):
    smb = pr[["S/L", "S/M", "S/H"]].mean(axis=1, skipna=False) - pr[["B/L", "B/M", "B/H"]].mean(axis=1, skipna=False)
    hml = 0.5 * (pr["S/H"] + pr["B/H"]) - 0.5 * (pr["S/L"] + pr["B/L"])
    return smb, hml


def stats(s):
    s = s.dropna()
    n = len(s)
    return {"months": n, "mean_pct": s.mean() * 100, "std_pct": s.std() * 100,
            "t_mean": s.mean() / (s.std() / np.sqrt(n)), "annualized_mean_pct": s.mean() * 1200}


def main():
    crsp = pd.read_parquet(PROC / "crsp_clean.parquet",
                           columns=["PERMNO", "ym", "MthRet", "lag_me", "ret_ok"])
    uni = pd.read_parquet(PROC / "formation_universe.parquet")
    ff = load_ff()

    # ---- return panel: one row per stock-month with a usable return and lagged ME ----
    r = crsp[crsp["ret_ok"] & crsp["ym"].between(FIRST_YM, LAST_YM)].copy()
    r["t"] = np.where(r["ym"] % 100 >= 7, r["ym"] // 100, r["ym"] // 100 - 1)   # formation year
    r["w"] = r["lag_me"]
    r["wr"] = r["lag_me"] * r["MthRet"]
    log(f"stock-months in the return panel: {len(r):,}; months: {r['ym'].nunique()}")

    # ---- market return: all eligible CRSP stocks (baseline) and the paper's narrower universe ----
    g = r.groupby("ym")[["w", "wr"]].sum()
    mkt = g["wr"] / g["w"]
    key = uni.loc[uni["has_be"], ["PERMNO", "t"]]
    rp = r.merge(key, on=["PERMNO", "t"])
    gp = rp.groupby("ym")[["w", "wr"]].sum()
    mkt_paper = gp["wr"] / gp["w"]

    # ---- baseline (security-level ME in BE/ME) and sensitivity (firm-level ME) ----
    a, bps = assign(uni, "bm_sec")
    pr, cnt = port_returns(r, a)
    smb, hml = factors(pr)
    a2, _ = assign(uni, "bm_firm")
    pr2, _ = port_returns(r, a2)
    smb2, hml2 = factors(pr2)

    f = pd.DataFrame({"mkt": mkt, "mkt_paper": mkt_paper, "smb": smb, "hml": hml,
                      "smb_firmME": smb2, "hml_firmME": hml2}).join(ff)
    f["mkt_rf"] = f["mkt"] - f["rf"]
    f["mkt_rf_paper"] = f["mkt_paper"] - f["rf"]
    f = f[(f.index >= FIRST_YM) & (f.index <= LAST_YM)]
    log(f"months in the factor series: {len(f)} ({f.index.min()} to {f.index.max()}); "
        f"months with a missing value: {int(f.isna().any(axis=1).sum())}")

    # ---- composition diagnostics (formation years) ----
    comp = a.groupby(["t", "port"]).agg(n=("PERMNO", "size"), me=("me_jun", "sum")).reset_index()
    comp["me_share"] = comp["me"] / comp.groupby("t")["me"].transform("sum")
    avg = comp.groupby("port")[["n", "me_share"]].mean().loc[PORTS]
    log("\n--- average composition across formation years ---")
    log(avg.rename(columns={"n": "avg_stocks", "me_share": "avg_share_of_ME"}).round(3).to_string())
    log(f"smallest number of stocks in any portfolio-year: {int(comp['n'].min())}")
    log(f"months with an empty portfolio: {int(pr.isna().any(axis=1).sum())}")

    # ---- summary statistics (monthly, percent) ----
    cols = {"MKT-RF (baseline)": "mkt_rf", "MKT-RF (paper universe)": "mkt_rf_paper", "SMB": "smb", "HML": "hml",
            "MKT-RF benchmark": "mkt_rf_b", "SMB benchmark": "smb_b", "HML benchmark": "hml_b"}
    st = pd.DataFrame({k: stats(f[v]) for k, v in cols.items()}).T.round(3)
    log("\n--- summary statistics (monthly, in percent; t = mean / (std / sqrt(T))) ---")
    log(st.to_string())

    log("\n--- correlations among my factors ---")
    log(f[["mkt_rf", "smb", "hml"]].corr().round(3).to_string())

    log("\n--- agreement with the French Data Library ---")
    for name, ours, bench in [("MKT-RF baseline", "mkt_rf", "mkt_rf_b"), ("MKT-RF paper universe", "mkt_rf_paper", "mkt_rf_b"),
                              ("SMB baseline", "smb", "smb_b"), ("SMB firm-level ME", "smb_firmME", "smb_b"),
                              ("HML baseline", "hml", "hml_b"), ("HML firm-level ME", "hml_firmME", "hml_b")]:
        d = f[[ours, bench]].dropna()
        slope, icpt = np.polyfit(d[bench], d[ours], 1)
        diff = d[ours] - d[bench]
        log(f"{name:24s} corr={d[ours].corr(d[bench]):.4f}  slope={slope:.3f}  intercept={icpt * 100:.3f} pct/month  "
            f"mean abs diff={diff.abs().mean() * 100:.3f} pct pts  mean diff={diff.mean() * 100:.3f} pct pts")

    # ---- save (aggregate series only) ----
    OUT.mkdir(exist_ok=True)
    f[["mkt_rf", "mkt_rf_paper", "smb", "hml", "smb_firmME", "hml_firmME", "rf",
       "mkt_rf_b", "smb_b", "hml_b"]].to_csv(OUT / "05_factors_monthly.csv")
    pr.to_csv(OUT / "05_portfolio_returns_monthly.csv")
    bps.to_csv(OUT / "05_breakpoints.csv", index=False)
    comp.to_csv(OUT / "05_portfolio_composition.csv", index=False)
    (OUT / "05_factors_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nsaved the output/05_* files")


if __name__ == "__main__":
    main()