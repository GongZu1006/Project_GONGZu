"""04_link_and_form.py - CCM link and the June portfolio-formation universe.

Run from the project root:  python3 src/04_link_and_form.py
Inputs : processed/crsp_clean.parquet, processed/compustat_be.parquet, CCM.csv under data/
Outputs: processed/formation_universe.parquet   derived from licensed data: do NOT commit
         output/04_formation_attrition.csv, output/04_formation_by_year.csv,
         output/04_formation_report.txt          aggregate counts only
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, PROC, OUT = ROOT / "data", ROOT / "processed", ROOT / "output"
FIRST_T, LAST_T = 2001, 2025          # formation years (returns July 2001 - December 2025)
LINES = []
LINK_TYPES = ["LC", "LU", "LS"]   # baseline was ["LC", "LU"]


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def load_ccm():
    path = next(iter(sorted(DATA.rglob("CCM.csv"))), None)
    if path is None:
        raise FileNotFoundError("CCM.csv not found under data/")
    ccm = pd.read_csv(path, usecols=["gvkey", "LPERMNO", "LINKTYPE", "LINKPRIM", "LINKDT", "LINKENDDT"],
                      dtype={"LINKTYPE": str, "LINKPRIM": str, "LINKDT": str, "LINKENDDT": str})
    log(f"CCM rows: {len(ccm):,}")
    ccm = ccm[ccm["LINKTYPE"].isin(LINK_TYPES) & ccm["LINKPRIM"].isin(["P", "C"])].copy()
    log(f"CCM rows with LINKTYPE in {LINK_TYPES} and LINKPRIM in P/C: {len(ccm):,}")
    ccm["LINKDT"] = pd.to_datetime(ccm["LINKDT"], errors="coerce")
    open_end = ccm["LINKENDDT"].eq("E")
    ccm["LINKENDDT"] = pd.to_datetime(ccm["LINKENDDT"].where(~open_end), errors="coerce")
    ccm.loc[open_end, "LINKENDDT"] = pd.Timestamp("2099-12-31")
    n_bad = int(ccm["LINKDT"].isna().sum() + ccm["LINKENDDT"].isna().sum())
    log(f"open-ended links (LINKENDDT = 'E'): {int(open_end.sum()):,}; unparsable dates dropped: {n_bad:,}")
    ccm = ccm.dropna(subset=["LINKDT", "LINKENDDT"])
    ccm["gvkey"] = ccm["gvkey"].astype("Int64")
    return ccm


def main():
    ccm = load_ccm()
    crsp = pd.read_parquet(PROC / "crsp_clean.parquet", columns=["PERMNO", "PERMCO", "ym", "me", "PrimaryExch"])
    be = pd.read_parquet(PROC / "compustat_be.parquet", columns=["gvkey", "cal_year", "datadate", "be"])
    be["gvkey"] = be["gvkey"].astype("Int64")

    # ---- 1. June universe: ME in June t and in December t-1 ----
    p = crsp[crsp["me"].notna()].copy()
    p["yr"], p["mon"] = p["ym"] // 100, p["ym"] % 100
    jun = (p[(p["mon"] == 6) & p["yr"].between(FIRST_T, LAST_T)]
           .rename(columns={"me": "me_jun", "yr": "t"}).drop(columns=["ym", "mon"]))
    dec = (p[(p["mon"] == 12) & p["yr"].between(FIRST_T - 1, LAST_T - 1)]
           .rename(columns={"me": "me_dec"}).assign(t=lambda d: d["yr"] + 1)
           .drop(columns=["ym", "mon", "yr"]))
    jun["me_jun_firm"] = jun.groupby(["PERMCO", "t"])["me_jun"].transform("sum")
    dec["me_dec_firm"] = dec.groupby(["PERMCO", "t"])["me_dec"].transform("sum")
    uni = jun.merge(dec[["PERMNO", "t", "me_dec", "me_dec_firm"]], on=["PERMNO", "t"], how="left")

    att = [("stock-years with ME in June t", len(uni), uni["PERMNO"].nunique())]
    uni = uni[uni["me_dec"].notna()].reset_index(drop=True)
    att.append(("... and ME in December t-1", len(uni), uni["PERMNO"].nunique()))
    uni["uid"] = uni.index
    uni["jdate"] = pd.to_datetime(uni["t"].astype(str) + "-06-30")

    # ---- 2. CCM link valid on June 30 of year t ----
    m = uni[["uid", "PERMNO", "jdate"]].merge(ccm, left_on="PERMNO", right_on="LPERMNO")
    m = m[(m["LINKDT"] <= m["jdate"]) & (m["jdate"] <= m["LINKENDDT"])]
    amb = int((m.groupby("uid")["gvkey"].nunique() > 1).sum())
    log(f"stock-years with more than one valid gvkey: {amb:,}")
    m["p_rank"] = (m["LINKPRIM"] != "P").astype(int)
    m["t_rank"] = (m["LINKTYPE"] != "LC").astype(int)
    m = m.sort_values(["uid", "p_rank", "t_rank", "LINKDT", "gvkey"],
                      ascending=[True, True, True, False, True])
    best = m.drop_duplicates("uid")[["uid", "gvkey", "LINKPRIM", "LINKTYPE"]]
    uni = uni.merge(best, on="uid", how="left")
    uni["has_link"] = uni["gvkey"].notna()

    # ---- 3. Book equity for the fiscal year ending in calendar year t-1 ----
    uni["fy"] = uni["t"] - 1
    be = be.rename(columns={"cal_year": "fy", "datadate": "be_datadate"})
    uni = uni.merge(be, on=["gvkey", "fy"], how="left")
    uni["has_be_row"] = uni["be_datadate"].notna()
    uni["has_be"] = uni["be"].notna()
    uni["be_pos"] = uni["be"] > 0
    uni["eligible"] = uni["has_link"] & uni["has_be"] & uni["be_pos"]
    uni["nyse"] = uni["PrimaryExch"] == "N"
    uni["bm_sec"] = np.where(uni["eligible"], uni["be"] / uni["me_dec"], np.nan)
    uni["bm_firm"] = np.where(uni["eligible"], uni["be"] / uni["me_dec_firm"], np.nan)

    def add(label, mask):
        att.append((label, int(mask.sum()), uni.loc[mask, "PERMNO"].nunique()))

    add("... with a valid CCM link on June 30", uni["has_link"])
    add("... and a Compustat record for the fiscal year ending in t-1", uni["has_link"] & uni["has_be_row"])
    add("... and computable book equity", uni["has_link"] & uni["has_be"])
    add("... and BE > 0 (eligible for the SMB/HML portfolios)", uni["eligible"])

    # ---- 4. diagnostics (aggregate numbers only) ----
    log("\n--- reasons for no BE among linked stock-years ---")
    linked = uni[uni["has_link"]]
    log(f"linked stock-years: {len(linked):,}")
    log(f"  no Compustat record for FY ending in t-1: {int((~linked['has_be_row']).sum()):,}")
    log(f"  record exists but BE not computable:     {int((linked['has_be_row'] & ~linked['has_be']).sum()):,}")
    log(f"  BE <= 0:                                 {int((linked['has_be'] & ~linked['be_pos']).sum()):,}")

    el = uni[uni["eligible"]]
    log("\n--- eligible share by exchange (PrimaryExch in June) ---")
    log(uni.groupby("PrimaryExch")["eligible"].agg(["size", "mean"]).rename(columns={"size": "stock_years", "mean": "eligible_share"}).round(3).to_string())

    multi = (el["me_dec_firm"] > el["me_dec"] * 1.0001)
    log(f"\neligible stock-years whose firm has several share classes (firm ME > security ME): {int(multi.sum()):,} ({multi.mean():.2%})")
    for c in ["bm_sec", "bm_firm"]:
        q = el[c].quantile([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
        log(f"{c} quantiles: " + ", ".join(f"p{int(k * 100)}={v:.3f}" for k, v in q.items()))

    g = uni.groupby("t")
    by = pd.DataFrame({
        "universe": g.size(),
        "linked": g["has_link"].sum(),
        "with_be_row": g["has_be_row"].sum(),
        "with_be": g["has_be"].sum(),
        "eligible": g["eligible"].sum(),
        "nyse_universe": g["nyse"].sum(),
    })
    by["nyse_eligible"] = el.groupby("t")["nyse"].sum()
    by["share_count"] = (by["eligible"] / by["universe"]).round(3)
    by["share_mcap"] = (el.groupby("t")["me_jun"].sum() / g["me_jun"].sum()).round(3)
    log("\n--- by formation year (June t) ---")
    log(by.to_string())

    # ---- 5. save aggregates first, then the licensed-data panel ----
    PROC.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    att_df = pd.DataFrame(att, columns=["step", "stock_years", "distinct_permno"])
    att_df.to_csv(OUT / "04_formation_attrition.csv", index=False)
    by.to_csv(OUT / "04_formation_by_year.csv")
    log("\n--- attrition (Table 2.1) ---")
    log(att_df.to_string(index=False))
    (OUT / "04_formation_report.txt").write_text("\n".join(LINES), encoding="utf-8")

    keep = ["PERMNO", "PERMCO", "gvkey", "t", "PrimaryExch", "nyse", "me_jun", "me_dec",
            "me_jun_firm", "me_dec_firm", "be", "bm_sec", "bm_firm", "has_link", "has_be_row",
            "has_be", "be_pos", "eligible", "LINKPRIM", "LINKTYPE"]
    uni[keep].to_parquet(PROC / "formation_universe.parquet", index=False)
    print("saved processed/formation_universe.parquet")


if __name__ == "__main__":
    main()