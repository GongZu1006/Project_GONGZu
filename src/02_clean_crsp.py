"""02_clean_crsp.py - Build the cleaned CRSP security-month panel.

Run from the project root:  python3 src/02_clean_crsp.py
Reads data/ only (never modifies it). Outputs:
  processed/crsp_clean.parquet     derived from licensed data: do NOT commit
  output/02_crsp_attrition.csv     aggregate counts only
  output/02_crsp_clean_report.txt  aggregate diagnostics only
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = ROOT / "processed"
OUT = ROOT / "output"
CHUNK = 500_000
KEY = ["PERMNO", "YYYYMM"]
LINES = []

FLAGS = ["MthDelFlg", "PrimaryExch", "SecurityType", "SecuritySubType", "ShareType",
         "USIncFlg", "IssuerType", "ConditionalType", "TradingStatusFlg"]
COLS = ["PERMNO", "PERMCO", "YYYYMM", "MthCalDt", "MthRet", "MthPrc", "ShrOut",
        "MthCap", "MthPrevCap", "vwretd"] + FLAGS

# Cumulative universe filters (Section 2.3, Step 1 of the report)
STEPS = [
    ("EQTY and COM", lambda d: (d["SecurityType"] == "EQTY") & (d["SecuritySubType"] == "COM")),
    ("ShareType NS", lambda d: d["ShareType"] == "NS"),
    ("US-incorporated", lambda d: d["USIncFlg"] == "Y"),
    ("IssuerType ACOR/CORP", lambda d: d["IssuerType"].isin(["ACOR", "CORP"])),
    ("Exchange N/A/Q", lambda d: d["PrimaryExch"].isin(["N", "A", "Q"])),
]


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def quantiles(s, label):
    q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    log(f"{label}: " + ", ".join(f"p{int(k * 100)}={v:.4f}" for k, v in q.items())
        + f"  | share within 0.1% of 1: {((s - 1).abs() < 0.001).mean():.2%}")


def main():
    path = next(iter(sorted(DATA.rglob("monthly_stock.csv"))), None)
    if path is None:
        raise FileNotFoundError("monthly_stock.csv not found under data/")
    log(f"reading {path.relative_to(ROOT)}")

    # ---- 1. chunked read + cumulative universe filters ----
    dtypes = {c: str for c in FLAGS}
    n_raw, perm_raw = 0, set()
    step_rows = [0] * len(STEPS)
    step_perm = [set() for _ in STEPS]
    kept = []
    for chunk in pd.read_csv(path, usecols=COLS, dtype=dtypes, chunksize=CHUNK):
        n_raw += len(chunk)
        perm_raw.update(chunk["PERMNO"].unique())
        d = chunk
        for i, (_, f) in enumerate(STEPS):
            d = d[f(d)]
            step_rows[i] += len(d)
            step_perm[i].update(d["PERMNO"].unique())
        kept.append(d)
    df = pd.concat(kept, ignore_index=True)
    del kept

    att = [("raw monthly_stock.csv", n_raw, len(perm_raw))]
    att += [(name, step_rows[i], len(step_perm[i])) for i, (name, _) in enumerate(STEPS)]

    # ---- 2. duplicates on (PERMNO, YYYYMM) ----
    dups = df[df.duplicated(KEY, keep=False)]
    if len(dups):
        conflict = (dups.groupby(KEY)[["MthRet", "MthPrc", "ShrOut"]]
                    .nunique(dropna=False).max(axis=1) > 1).sum()
        log(f"duplicate rows on {KEY}: {len(dups):,}; keys with conflicting values: {conflict:,}")
    else:
        log(f"no duplicates on {KEY}")
    df = df.drop_duplicates(KEY, keep="first").copy()
    att.append(("after dropping duplicates", len(df), df["PERMNO"].nunique()))

    # ---- 3. dates, ME, lagged ME ----
    df["date"] = pd.to_datetime(df["MthCalDt"])
    df["ym"] = df["YYYYMM"].astype(int)
    mismatch = (df["date"].dt.year * 100 + df["date"].dt.month != df["ym"]).sum()
    log(f"rows where MthCalDt and YYYYMM disagree: {mismatch:,}")

    df["me"] = df["MthPrc"].abs() * df["ShrOut"]            # $ thousands
    df.loc[(df["ShrOut"] <= 0) | (df["me"] <= 0), "me"] = np.nan

    df = df.sort_values(["PERMNO", "ym"]).reset_index(drop=True)
    df["mi"] = (df["ym"] // 100) * 12 + df["ym"] % 100       # month counter
    g = df.groupby("PERMNO")
    df["lag_me"] = g["me"].shift(1).where(df["mi"] - g["mi"].shift(1) == 1)

    df["has_ret"] = df["MthRet"].notna()
    df["has_me"] = df["me"].notna()
    df["has_lag_me"] = df["lag_me"].notna()
    df["ret_ok"] = df["has_ret"] & df["has_lag_me"]
    att.append(("has return (flag)", int(df["has_ret"].sum()), df.loc[df["has_ret"], "PERMNO"].nunique()))
    att.append(("has ME (flag)", int(df["has_me"].sum()), df.loc[df["has_me"], "PERMNO"].nunique()))
    att.append(("has return and lagged ME (flag)", int(df["ret_ok"].sum()), df.loc[df["ret_ok"], "PERMNO"].nunique()))

    # ---- 4. diagnostics (aggregate numbers only) ----
    log("\n--- units / consistency checks ---")
    both = df["me"].notna() & (df["MthCap"] > 0)
    quantiles(df.loc[both, "me"] / df.loc[both, "MthCap"], "ME / MthCap")
    both = df["lag_me"].notna() & (df["MthPrevCap"] > 0)
    quantiles(df.loc[both, "lag_me"] / df.loc[both, "MthPrevCap"], "lag_me / MthPrevCap")
    log(f"share of rows with negative MthPrc: {(df['MthPrc'] < 0).mean():.2%}")

    log("\n--- TradingStatusFlg x MthDelFlg (rows) ---")
    log(pd.crosstab(df["TradingStatusFlg"].fillna("<NA>"), df["MthDelFlg"].fillna("<NA>")).to_string())
    log("\n--- ConditionalType x MthDelFlg (rows) ---")
    log(pd.crosstab(df["ConditionalType"].fillna("<NA>"), df["MthDelFlg"].fillna("<NA>")).to_string())
    log("\nshare of missing MthRet by MthDelFlg:")
    log(df.groupby(df["MthDelFlg"].fillna("<NA>"))["MthRet"].apply(lambda s: s.isna().mean()).round(4).to_string())

    log("\n--- stocks per month ---")
    cnt = df.groupby("ym").size()
    log(f"months: {len(cnt)}; min={cnt.min():,}; median={int(cnt.median()):,}; max={cnt.max():,}")
    log("average number of stocks per month, by year:")
    log(cnt.groupby(cnt.index // 100).mean().round(0).astype(int).to_string())
    log("rows by PrimaryExch: " + df["PrimaryExch"].value_counts().to_dict().__str__())

    log("\n--- my value-weighted return vs CRSP vwretd ---")
    d = df[df["ret_ok"]]
    agg = pd.DataFrame({"w": d["lag_me"], "wr": d["lag_me"] * d["MthRet"], "ym": d["ym"]}).groupby("ym").sum()
    ours = (agg["wr"] / agg["w"]).rename("ours")
    bench = df.groupby("ym")["vwretd"].first()
    comp = pd.concat([ours, bench], axis=1).dropna()
    log(f"months compared: {len(comp)}; correlation: {comp['ours'].corr(comp['vwretd']):.4f}; "
        f"mean abs difference: {(comp['ours'] - comp['vwretd']).abs().mean() * 100:.3f} pct points; "
        f"mean difference (ours - vwretd): {(comp['ours'] - comp['vwretd']).mean() * 100:.3f} pct points")

    # ---- 5. save (aggregate outputs first, then the licensed-data panel) ----
    PROC.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    att_df = pd.DataFrame(att, columns=["step", "rows", "distinct_permno"])
    att_df.to_csv(OUT / "02_crsp_attrition.csv", index=False)
    log("\n--- attrition ---")
    log(att_df.to_string(index=False))
    (OUT / "02_crsp_clean_report.txt").write_text("\n".join(LINES), encoding="utf-8")

    keep = ["PERMNO", "PERMCO", "ym", "date", "MthRet", "MthPrc", "ShrOut", "MthCap",
            "me", "lag_me", "MthDelFlg", "PrimaryExch", "ConditionalType",
            "TradingStatusFlg", "has_ret", "has_me", "has_lag_me", "ret_ok"]
    df[keep].to_parquet(PROC / "crsp_clean.parquet", index=False)
    print("saved processed/crsp_clean.parquet")


if __name__ == "__main__":
    main()