"""08_compare_breakpoints.py - Compare my June NYSE breakpoints with the French Data Library files.

Run from the project root:  python3 src/08_compare_breakpoints.py
Inputs : data/ME_Breakpoints.csv and data/BE-ME_Breakpoints.csv (public library files, found with rglob),
         output/05_breakpoints.csv, processed/crsp_clean.parquet, processed/formation_universe.parquet
Outputs: output/08_breakpoints_compare.csv and output/08_breakpoints_report.txt (aggregate numbers only)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, PROC, OUT = ROOT / "data", ROOT / "processed", ROOT / "output"
FIRST_T, LAST_T = 2001, 2025
PCTS = list(range(5, 105, 5))            # 5, 10, ..., 100 (20 values)
LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def find(name):
    hits = sorted(DATA.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under data/")
    return hits[0]


def numbers(line):
    return [float(x) for x in line.split(",") if x.strip() != ""]


def read_me(path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"^\s*\d{6}\s*,", line):
            v = numbers(line)
            rows[int(v[0])] = v[1:]
    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.shape[1] != 21:
        raise ValueError(f"ME file: expected 21 columns after the date, got {df.shape[1]}")
    df.columns = ["n"] + [f"p{p}" for p in PCTS]
    return df


def read_beme(path):
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"^\s*\d{4}\s*,", line):
            v = numbers(line)
            rows[int(v[0])] = v[1:]
    df = pd.DataFrame.from_dict(rows, orient="index")
    if df.shape[1] != 22:
        raise ValueError(f"BE/ME file: expected 22 columns after the year, got {df.shape[1]}")
    df.columns = ["n_le0", "n_gt0"] + [f"p{p}" for p in PCTS]
    return df


def main():
    t_idx = list(range(FIRST_T, LAST_T + 1))
    me_lib = read_me(find("ME_Breakpoints.csv"))
    be_lib = read_beme(find("BE-ME_Breakpoints.csv"))
    log(f"library ME file: {len(me_lib)} months ({me_lib.index.min()} to {me_lib.index.max()}); "
        f"BE/ME file: {len(be_lib)} years ({be_lib.index.min()} to {be_lib.index.max()})")

    mine = pd.read_csv(OUT / "05_breakpoints.csv").set_index("t").loc[t_idx]

    # my NYSE stocks in June, all of them with market equity (the paper's "all NYSE stocks on CRSP")
    crsp = pd.read_parquet(PROC / "crsp_clean.parquet", columns=["PERMNO", "ym", "me", "PrimaryExch"])
    j = crsp[(crsp["ym"] % 100 == 6) & crsp["me"].notna() & (crsp["PrimaryExch"] == "N")].copy()
    j["t"] = j["ym"] // 100
    allny = j.groupby("t")["me"].agg(n_all="size", med_all="median").loc[t_idx]

    lib_me = me_lib.loc[[t * 100 + 6 for t in t_idx]]
    lib_me.index = t_idx

    # ---- 1. size breakpoint (median NYSE ME in June) ----
    c = pd.DataFrame(index=t_idx)
    c["lib_median_musd"] = lib_me["p50"]
    c["mine_allNYSE_musd"] = allny["med_all"] / 1000.0           # my ME is in $ thousands
    c["mine_eligNYSE_musd"] = mine["size_median_ME"] / 1000.0
    c["ratio_allNYSE"] = c["mine_allNYSE_musd"] / c["lib_median_musd"]
    c["ratio_eligNYSE"] = c["mine_eligNYSE_musd"] / c["lib_median_musd"]
    c["n_lib_nyse"] = lib_me["n"]
    c["n_mine_allNYSE"] = allny["n_all"]
    log("\n--- size breakpoint: median NYSE ME in June ($ millions if the library unit is as assumed) ---")
    log(f"average ratio mine(all NYSE stocks)/library:      {c['ratio_allNYSE'].mean():.4f}   "
        f"mean |ratio-1|: {(c['ratio_allNYSE'] - 1).abs().mean():.4f}")
    log(f"average ratio mine(eligible NYSE stocks)/library: {c['ratio_eligNYSE'].mean():.4f}   "
        f"mean |ratio-1|: {(c['ratio_eligNYSE'] - 1).abs().mean():.4f}")
    log(f"average ratio of NYSE stock counts, mine(all)/library: {(c['n_mine_allNYSE'] / c['n_lib_nyse']).mean():.4f}")

    # ---- 2. BE/ME breakpoints: find the year alignment that fits best ----
    errs = {}
    for s in (-1, 0, 1):
        lib = be_lib.reindex([t + s for t in t_idx])
        lib.index = t_idx
        e30 = (mine["bm_p30"] / lib["p30"] - 1).abs().mean()
        e70 = (mine["bm_p70"] / lib["p70"] - 1).abs().mean()
        errs[s] = e30 + e70
        log(f"BE/ME file row = formation year + {s}: mean |relative diff| p30 {e30:.4f}, p70 {e70:.4f}")
    best = min(errs, key=errs.get)
    log(f"best alignment: row = formation year + {best}")
    lib_be = be_lib.reindex([t + best for t in t_idx])
    lib_be.index = t_idx
    c["lib_bm_p30"], c["mine_bm_p30"] = lib_be["p30"], mine["bm_p30"]
    c["lib_bm_p70"], c["mine_bm_p70"] = lib_be["p70"], mine["bm_p70"]
    c["ratio_p30"] = c["mine_bm_p30"] / c["lib_bm_p30"]
    c["ratio_p70"] = c["mine_bm_p70"] / c["lib_bm_p70"]
    c["n_lib_nyse_BEpos"] = lib_be["n_gt0"]
    c["n_mine_nyse_eligible"] = mine["n_nyse"]
    log(f"average ratio mine/library: p30 {c['ratio_p30'].mean():.4f}, p70 {c['ratio_p70'].mean():.4f}")
    log(f"average ratio of NYSE stocks with BE>0: mine(eligible)/library {(c['n_mine_nyse_eligible'] / c['n_lib_nyse_BEpos']).mean():.4f}")

    # ---- 3. how many eligible stocks would change group under the library's breakpoints ----
    uni = pd.read_parquet(PROC / "formation_universe.parquet", columns=["t", "eligible", "me_jun", "bm_sec"])
    el = uni[uni["eligible"]]
    rows = []
    for t in t_idx:
        d = el[el["t"] == t]

        def between(x, a, b):
            return int(((x > min(a, b)) & (x <= max(a, b))).sum())

        rows.append([t, len(d),
                     between(d["me_jun"], mine.loc[t, "size_median_ME"], lib_me.loc[t, "p50"] * 1000.0),
                     between(d["bm_sec"], mine.loc[t, "bm_p30"], lib_be.loc[t, "p30"]),
                     between(d["bm_sec"], mine.loc[t, "bm_p70"], lib_be.loc[t, "p70"])])
    sw = pd.DataFrame(rows, columns=["t", "eligible", "switch_size", "switch_p30", "switch_p70"]).set_index("t")
    sw["share_switching_any"] = ((sw["switch_size"] + sw["switch_p30"] + sw["switch_p70"]) / sw["eligible"]).round(4)
    log("\n--- eligible stocks that would change group if the library's breakpoints were used ---")
    log(sw.to_string())
    log(f"average share of stocks affected by at least one breakpoint (upper bound): {sw['share_switching_any'].mean():.2%}")

    out = c.join(sw)
    out.round(4).to_csv(OUT / "08_breakpoints_compare.csv")
    log("\n--- full comparison table ---")
    log(c.round(3).to_string())
    (OUT / "08_breakpoints_report.txt").write_text("\n".join(LINES), encoding="utf-8")
    log("\nsaved output/08_breakpoints_compare.csv and 08_breakpoints_report.txt")


if __name__ == "__main__":
    main()