"""01_explore.py - Read-only inspection of the raw data files.

Run from the project root:  python src/01_explore.py
Writes a text summary (counts, ranges, value counts; no raw data rows) to
output/01_data_overview.txt
"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output" / "01_data_overview.txt"
CHUNK = 500_000
LINES = []


def log(msg=""):
	print(msg)
	LINES.append(str(msg))


def find_file(name):
	hits = sorted(DATA.rglob(name))
	if not hits:
		log(f"!! {name} not found under {DATA}")
		return None
	if len(hits) > 1:
		log(f"!! multiple matches for {name}: using {hits[0]}")
	return hits[0]


def describe_header(path):
	log(f"\n=== {path.name} ===")
	log(f"path: {path.relative_to(ROOT)}")
	log(f"size: {path.stat().st_size / 1e6:,.1f} MB")
	sample = pd.read_csv(path, nrows=100_000)
	log("columns and dtypes (100,000-row sample):")
	for col, dt in sample.dtypes.items():
		log(f"  {col}: {dt}")


def scan(path, cols, flag_cols=(), date_cols=(), missing_cols=(), id_col=None, year_col=None):
	"""Chunked scan. Only the requested columns are read; all read as text."""
	available = set(pd.read_csv(path, nrows=0).columns)
	absent = [c for c in cols if c not in available]
	if absent:
		log(f"!! columns listed in the dictionary but absent from the file: {absent}")
	use = [c for c in cols if c in available]

	n_rows = 0
	flags = {c: Counter() for c in flag_cols if c in available}
	dmin, dmax = {}, {}
	bad_count = Counter()
	bad_examples = {c: set() for c in date_cols if c in available}
	miss = Counter()
	ids = set()
	by_year = Counter()

	for chunk in pd.read_csv(path, usecols=use, dtype=str, chunksize=CHUNK):
		n_rows += len(chunk)
		for c in flags:
			flags[c].update(chunk[c].fillna("<NA>").value_counts().to_dict())
		for c in bad_examples:
			parsed = pd.to_datetime(chunk[c], errors="coerce")
			bad = parsed.isna() & chunk[c].notna()
			bad_count[c] += int(bad.sum())
			if len(bad_examples[c]) < 5:
				bad_examples[c].update(chunk.loc[bad, c].unique()[:5])
			if parsed.notna().any():
				lo, hi = parsed.min(), parsed.max()
				dmin[c] = lo if c not in dmin else min(dmin[c], lo)
				dmax[c] = hi if c not in dmax else max(dmax[c], hi)
			if c == year_col:
				by_year.update(parsed.dt.year.dropna().astype(int).value_counts().to_dict())
		for c in missing_cols:
			if c in chunk:
				miss[c] += int(chunk[c].isna().sum())
		if id_col in chunk:
			ids.update(chunk[id_col].dropna().unique())

	log(f"rows: {n_rows:,}")
	if id_col:
		log(f"distinct {id_col}: {len(ids):,}")
	for c in bad_examples:
		log(f"{c}: min={dmin.get(c)}  max={dmax.get(c)}  "
			f"non-date values={bad_count[c]:,}  examples={sorted(bad_examples[c])}")
	if by_year:
		log("rows per year: " + ", ".join(f"{y}:{k:,}" for y, k in sorted(by_year.items())))
	for c, ctr in flags.items():
		log(f"value counts for {c}:")
		for val, k in ctr.most_common(30):
			log(f"  {val!r}: {k:,}")
	for c in missing_cols:
		if c in available and n_rows:
			log(f"missing share {c}: {miss[c] / n_rows:.2%}")


def inspect_ff(path):
	log(f"\n=== {path.name} ===")
	log(f"path: {path.relative_to(ROOT)}")
	raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
	log(f"total lines: {len(raw)}")
	log("first 15 lines (introductory text of the public library file):")
	for line in raw[:15]:
		log(f"  | {line}")
	pat = re.compile(r"^\s*(\d{6})\s*,")
	idx = [i for i, line in enumerate(raw) if pat.match(line)]
	if not idx:
		log("!! no YYYYMM rows found; the layout differs from what is expected")
		return
	keys = [pat.match(raw[i]).group(1) for i in idx]
	contiguous = idx[-1] - idx[0] + 1 == len(idx)
	log(f"monthly rows: {len(idx)} (line {idx[0] + 1} to line {idx[-1] + 1}); contiguous block: {contiguous}")
	log(f"first month: {keys[0]}   last month: {keys[-1]}")
	log(f"column header line just above the monthly block: {raw[idx[0] - 1]!r}")


def main():
	log(f"project root: {ROOT}")
	log(f"data folder:  {DATA}")

	p = find_file("monthly_stock.csv")
	if p:
		describe_header(p)
		scan(
			p,
			cols=["PERMNO", "PERMCO", "MthCalDt", "MthRet", "MthRetx", "MthPrc", "ShrOut",
				  "MthCap", "MthPrevCap", "MthPrevDt", "MthDelFlg", "SecInfoStartDt",
				  "SecInfoEndDt", "PrimaryExch", "ShareType", "SecurityType",
				  "SecuritySubType", "USIncFlg", "IssuerType", "ConditionalType",
				  "TradingStatusFlg"],
			flag_cols=["PrimaryExch", "SecurityType", "SecuritySubType", "ShareType",
					   "USIncFlg", "IssuerType", "ConditionalType", "TradingStatusFlg",
					   "MthDelFlg"],
			date_cols=["MthCalDt"],
			missing_cols=["MthRet", "MthPrc", "ShrOut", "MthCap"],
			id_col="PERMNO",
			year_col="MthCalDt",
		)

	p = find_file("Compustat.csv")
	if p:
		describe_header(p)
		scan(
			p,
			cols=["gvkey", "datadate", "indfmt", "datafmt", "consol", "curcd", "pstkrv",
				  "pstkl", "pstk", "seq", "ceq", "at", "lt", "txditc"],
			flag_cols=["indfmt", "datafmt", "consol", "curcd"],
			date_cols=["datadate"],
			missing_cols=["seq", "ceq", "at", "lt", "txditc", "pstkrv", "pstkl", "pstk"],
			id_col="gvkey",
			year_col="datadate",
		)

	p = find_file("CCM.csv")
	if p:
		describe_header(p)
		scan(
			p,
			cols=["gvkey", "LPERMNO", "LINKTYPE", "LINKPRIM", "LINKDT", "LINKENDDT"],
			flag_cols=["LINKTYPE", "LINKPRIM"],
			date_cols=["LINKDT", "LINKENDDT"],
			missing_cols=["LINKDT", "LINKENDDT"],
			id_col="LPERMNO",
		)

	p = find_file("F-F factors and RF.csv")
	if p:
		inspect_ff(p)

	OUT.parent.mkdir(parents=True, exist_ok=True)
	OUT.write_text("\n".join(LINES), encoding="utf-8")
	log(f"\nsummary written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
	main()
