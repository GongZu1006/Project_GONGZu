"""03_build_be.py - Book equity from Compustat annual data.

Run from the project root:  python3 src/03_build_be.py
Reads data/ only (never modifies it). Outputs:
  processed/compustat_be.parquet    derived from licensed data: do NOT commit
  output/03_be_attrition.csv        aggregate counts only
  output/03_be_by_year.csv          aggregate counts only
  output/03_be_report.txt           aggregate diagnostics only
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = ROOT / "processed"
OUT = ROOT / "output"
LINES = []

COLS = ["gvkey", "datadate", "fyear", "indfmt", "datafmt", "consol", "curcd",
        "at", "ceq", "lt", "pstk", "pstkl", "pstkrv", "seq", "txditc"]
NUM = ["at", "ceq", "lt", "pstk", "pstkl", "pstkrv", "seq", "txditc"]


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def main():
    path = next(iter(sorted(DATA.rglob("Compustat.csv"))), None)
    if path is None:
        raise FileNotFoundError("Compustat.csv not found under data/")
    log(f"reading {path.relative_to(ROOT)}")

    raw = pd.read_csv(path, usecols=COLS, dtype={c: str for c in ["indfmt", "datafmt", "consol", "curcd"]})
    att = [("raw Compustat.csv", len(raw), raw["gvkey"].nunique())]

    # ---- 1. standard industrial, consolidated, standardized, USD records ----
    df = raw[(raw["indfmt"] == "INDL") & (raw["datafmt"] == "STD") & (raw["consol"] == "C")]
    att.append(("INDL / STD / C", len(df), df["gvkey"].nunique()))
    df = df[df["curcd"] == "USD"].copy()
    att.append(("USD reporting currency", len(df), df["gvkey"].nunique()))

    # ---- 2. one record per gvkey-datadate, then one per gvkey-calendar year ----
    df["datadate"] = pd.to_datetime(df["datadate"])
    n_dup = int(df.duplicated(["gvkey", "datadate"]).sum())
    log(f"duplicate gvkey-datadate records dropped: {n_dup:,}")
    df = df.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "datadate"], keep="last")
    df["cal_year"] = df["datadate"].dt.year
    n_multi = int(df.duplicated(["gvkey", "cal_year"]).sum())
    log(f"records dropped because the firm has a later fiscal year-end in the same calendar year: {n_multi:,}")
    df = df.drop_duplicates(["gvkey", "cal_year"], keep="last").reset_index(drop=True)
    att.append(("one record per firm and calendar year", len(df), df["gvkey"].nunique()))

    log("\n--- coverage of raw variables (firm-years, USD) ---")
    empty = df[["seq", "ceq", "at", "lt"]].isna().all(axis=1)
    log(f"firm-years with seq, ceq, at and lt all missing: {int(empty.sum()):,} ({empty.mean():.2%})")
    for c in NUM:
        log(f"missing share {c}: {df[c].isna().mean():.2%}")

    # ---- 3. stockholders' equity with fallbacks ----
    se = df["seq"].copy()
    src = pd.Series("seq", index=df.index)
    src[se.isna()] = ""
    m = se.isna() & df["ceq"].notna()
    se[m] = df.loc[m, "ceq"] + df.loc[m, "pstk"].fillna(0)
    src[m] = "ceq+pstk"
    m = se.isna() & df["at"].notna() & df["lt"].notna()
    se[m] = df.loc[m, "at"] - df.loc[m, "lt"]
    src[m] = "at-lt"
    src[se.isna()] = "none"
    df["se_source"] = src

    # ---- 4. preferred stock, deferred taxes, book equity (in $ thousands) ----
    ps = df["pstkrv"].fillna(df["pstkl"]).fillna(df["pstk"]).fillna(0)
    tx = df["txditc"].fillna(0)
    be_m = se + tx - ps                       # $ millions
    df["be"] = be_m * 1000                    # $ thousands, same unit as ME
    df["txditc_missing"] = df["txditc"].isna()

    n_se = int(se.notna().sum())
    n_be = int(df["be"].notna().sum())
    n_pos = int((df["be"] > 0).sum())
    att.append(("stockholders' equity computable", n_se, df.loc[se.notna(), "gvkey"].nunique()))
    att.append(("book equity computed", n_be, df.loc[df["be"].notna(), "gvkey"].nunique()))
    att.append(("book equity > 0", n_pos, df.loc[df["be"] > 0, "gvkey"].nunique()))

    log("\n--- source of stockholders' equity ---")
    log(df["se_source"].value_counts().to_string())

    log("\n--- book equity ($ millions) ---")
    q = be_m.dropna().quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    log("quantiles: " + ", ".join(f"p{int(k * 100)}={v:,.1f}" for k, v in q.items()))
    log(f"share of computed BE that is <= 0: {(be_m.dropna() <= 0).mean():.2%}")
    both = be_m.notna() & df["at"].notna()
    log(f"firm-years with BE greater than total assets: {int((be_m[both] > df.loc[both, 'at']).sum()):,}")

    # ---- 5. by calendar year ----
    by = pd.DataFrame({
        "firm_years": df.groupby("cal_year").size(),
        "with_be": df.groupby("cal_year")["be"].apply(lambda s: int(s.notna().sum())),
        "be_positive": df.groupby("cal_year")["be"].apply(lambda s: int((s > 0).sum())),
    })
    with_be = df[df["be"].notna()]
    by["txditc_missing_share"] = with_be.groupby("cal_year")["txditc_missing"].mean().round(3)
    by["median_be_musd"] = (with_be.groupby("cal_year")["be"].median() / 1000).round(1)
    log("\n--- by calendar year of the fiscal year-end ---")
    log(by.to_string())

    # ---- 6. save aggregates first, then the licensed-data panel ----
    PROC.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    att_df = pd.DataFrame(att, columns=["step", "rows", "distinct_gvkey"])
    att_df.to_csv(OUT / "03_be_attrition.csv", index=False)
    by.to_csv(OUT / "03_be_by_year.csv")
    log("\n--- attrition ---")
    log(att_df.to_string(index=False))
    (OUT / "03_be_report.txt").write_text("\n".join(LINES), encoding="utf-8")

    keep = ["gvkey", "datadate", "cal_year", "fyear", "be", "se_source", "txditc_missing"]
    df[keep].to_parquet(PROC / "compustat_be.parquet", index=False)
    print("saved processed/compustat_be.parquet")


if __name__ == "__main__":
    main()