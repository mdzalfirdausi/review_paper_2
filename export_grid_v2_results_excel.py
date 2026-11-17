#!/usr/bin/env python3
"""Export Lens/Overton BERTopic passing configurations to a Scopus-style Excel workbook.

Run from SLR_Energy_Forecasting_and_OPF project root:
    python export_grid_v2_results_excel.py

Reads two local ranking CSVs. Keeps only rows with a finite Rank, which the
current ranking script uses to identify passing / eligible configurations.
Never invents missing evaluation metrics or stability scores.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from collections import Counter

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo
except ImportError as exc:
    raise SystemExit("Missing openpyxl. Install with: python -m pip install openpyxl") from exc

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "Results" / "bertopic_tuning"
OUTPUT = ROOT / "Results" / "bertopic_tuning" / "GRID_V2_RESULTS_LENS_OVERTON.xlsx"
CORPUS_DOCS = {"lens": 3470, "overton": 2728}

# Match the organizational style of the uploaded Scopus GRID_V2_RESULTS.xlsx.
# Some original Scopus metrics (e.g. stability) will be blank until computed.
SCOPUS_HEADERS = [
    "Rank", "Model", "n_neighbors", "n_components", "min_cluster_size_pct",
    "min_cluster_size_papers", "min_samples", "Selection", "Topics",
    "Collapsed (largest >= 50%)", "Outlier value %", "Outlier pts",
    "Outlier contrib", "Kappa value", "Kappa pts", "Kappa contrib",
    "NPMI value", "NPMI pts", "NPMI contrib", "Diversity value %",
    "Diversity pts", "Diversity contrib", "Coverage value", "Coverage pts",
    "Coverage contrib", "Cohesion value", "Cohesion pts", "Cohesion contrib",
    "Size value", "Size pts", "Size contrib", "Stability value",
    "Stability pts", "Stability contrib", "Seed topics (0/1/2)",
    "G1", "G2", "G3", "Score", "Mean rank (tie-break)", "Grid_id",
]

# Keep explicit semantic diagnostic after the Scopus base columns.
ADDITIONAL = ["Semantic Silhouette value", "Largest topic % (raw)"]

GRID_PATTERN = re.compile(
    r"^(mpnet|biobert_sts|minilm)_nn(\d+)_nc(\d+)_mcs([\d.]+)_ms(\d+)_(eom|leaf)$"
)
DISPLAY_MODELS = {
    "mpnet": "paraphrase-mpnet-base-v2",
    "biobert_sts": "BioBERT-STS",
    "minilm": "all-MiniLM-L6-v2",
}


def numeric(s):
    """Convert CSV number to int/float or None; never turn NaN into a number."""
    if s is None or str(s).strip().lower() in ("", "nan", "none", "null"):
        return None
    try:
        f = float(str(s).strip())
        if not math.isfinite(f):
            return None
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return None


def normalized(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def lookup(row, *names):
    by_key = {normalized(k): v for k, v in row.items()}
    for name in names:
        value = by_key.get(normalized(name))
        if value is not None and str(value).strip().lower() not in ("", "nan", "none"):
            return value
    return None


def number(row, *names):
    return numeric(lookup(row, *names))


def load(db):
    path = BASE / db / "scopus_aligned_grid_v2" / "tables" / f"{db}_ranked_pre.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Ranking CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        all_rows = list(reader)
    # rank is finite only for configurations passing the evaluator's screening.
    passing = [r for r in all_rows if number(r, "Rank") is not None]
    passing.sort(key=lambda r: (number(r, "Rank"), str(lookup(r, "Grid_id") or "")))
    print(f"{db.upper()}: {len(passing)} passing of {len(all_rows)} ranked rows; source={path}")
    return passing, columns


def parse_grid(row):
    grid_id = lookup(row, "Grid_id")
    m = GRID_PATTERN.fullmatch(str(grid_id or ""))
    if not m:
        return {}, grid_id
    model, nn, nc, mcs, ms, selection = m.groups()
    return {
        "Model": DISPLAY_MODELS[model],
        "n_neighbors": int(nn),
        "n_components": int(nc),
        "min_cluster_size_pct": float(mcs),
        "min_samples": int(ms),
        "Selection": selection,
    }, grid_id


def data_record(row, db, extra_source_columns):
    parsed, grid_id = parse_grid(row)
    d = dict(parsed)
    d["Grid_id"] = grid_id
    d["Rank"] = number(row, "Rank")
    d["Topics"] = number(row, "Topics", "n_topics", "topic_count")
    d["Outlier value %"] = number(row, "Outlier_pct", "outlier_percentage", "outlier_rate_pct")
    d["Score"] = number(row, "Weighted_score", "score", "pre_score")
    d["Mean rank (tie-break)"] = number(row, "Mean_rank", "mean_rank_tiebreak")

    mcs = parsed.get("min_cluster_size_pct")
    if mcs is not None:
        d["min_cluster_size_papers"] = max(10, round(mcs * CORPUS_DOCS[db] / 100))

    d["Kappa value"] = number(row, "Kappa", "kappa_value", "label_match_kappa")
    d["NPMI value"] = number(row, "NPMI", "npmi_value", "coherence_npmi")
    d["Diversity value %"] = number(row, "Diversity_pct", "topic_diversity_pct", "diversity_value")
    d["Coverage value"] = number(row, "Coverage", "label_coverage", "coverage_value")
    d["Cohesion value"] = number(row, "Cohesion", "topic_cohesion", "cohesion_value")
    d["Size value"] = number(row, "Size", "size_value", "largest_topic_pct", "largest_topic_percentage")
    d["Stability value"] = number(row, "Stability", "stability_ari", "seed_ari")
    d["Semantic Silhouette value"] = number(
        row, "Semantic_silhouette", "semantic_silhouette_score", "silhouette_semantic"
    )
    d["Largest topic % (raw)"] = number(
        row, "Largest_topic_pct", "largest_topic_percentage", "largest_topic_percent"
    )
    largest = d["Largest topic % (raw)"]
    d["Collapsed (largest >= 50%)"] = ("yes" if largest >= 50 else "no") if largest is not None else None
    # Respect original CSV point/contribution numbers if present; do not recalculate
    # and accidentally introduce a weighting system that differs from Scopus.
    for label, prefixes in {
        "Outlier": ["outlier"], "Kappa": ["kappa"], "NPMI": ["npmi"],
        "Diversity": ["diversity"], "Coverage": ["coverage"],
        "Cohesion": ["cohesion"], "Size": ["size"], "Stability": ["stability"],
    }.items():
        d[f"{label} pts"] = number(row, *(f"{p}_pts" for p in prefixes))
        d[f"{label} contrib"] = number(row, *(f"{p}_contrib" for p in prefixes))
    for group in ("G1", "G2", "G3"):
        d[group] = number(row, group)
    d["Seed topics (0/1/2)"] = lookup(row, "Seed_topics", "seed_topics_0_1_2")
    for c in extra_source_columns:
        d[f"CSV: {c}"] = numeric(row.get(c)) if numeric(row.get(c)) is not None else row.get(c)
    return d


def style_sheet(ws, headers, rows, title):
    ws.append(headers)
    for d in rows:
        ws.append([d.get(c) for c in headers])
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "C2"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.sheet_view.zoomScale = 80
    header_fill = PatternFill("solid", fgColor="183153")
    header_font = Font(name="Aptos", size=10, bold=True, color="FFFFFF")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 44
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, len(rows)+1)}"
    for i, col in enumerate(headers, start=1):
        letter = get_column_letter(i)
        width = (34 if col.startswith("CSV:") else 28 if col in ("Model", "Grid_id")
                 else 26 if len(col)>24 else 18 if len(col)>14 else 14)
        ws.column_dimensions[letter].width = width
    rank_i = headers.index("Rank") + 1
    score_i = headers.index("Score") + 1
    for row in ws.iter_rows(min_row=2):
        if row[0].row % 2 == 0:
            for c in row:
                c.fill = PatternFill("solid", fgColor="F0F5FA")
        row[rank_i-1].font = Font(name="Aptos", color="153C60", bold=True)
        row[score_i-1].font = Font(name="Aptos", color="087569", bold=True)
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.0000" if abs(cell.value) < 1 else "0.00"
    ws.print_options.horizontalCentered = True
    ws.print_title_rows = "1:1"


def main():
    wb = Workbook()
    notes = wb.active
    notes.title = "Notes"
    note_rows = [
        ["GRID V2 RESULTS — Lens & Overton"],
        ["Generated from the existing preliminary ranking CSVs on the ISS cluster."],
        ["Source: Results/bertopic_tuning/{lens,overton}/scopus_aligned_grid_v2/tables/{db}_ranked_pre.csv"],
        ["PASSING = rows with a finite Rank in the current evaluator output."],
        ["Original screening: 5–30 topics; outliers < 10%; largest topic < 50%."],
        ["This workbook is PRE-STABILITY and uses the evaluator's existing weights."],
        ["Column layout follows the uploaded GRID_V2_RESULTS.xlsx example."],
        ["Original Scopus metric names retained; missing or uncomputed metrics stay blank."],
        ["CSV: columns preserve every source field without replacing raw evaluator values."],
        ["Scopus evaluation methodology/weights may differ from current Lens/Overton evaluator."],
        ["Do not compare final scores across corpora before harmonizing metric definitions and weights."],
        ["Semantic Topic Redundancy is proposed but not yet computed; not added as a fabricated value."],
    ]
    for row in note_rows:
        notes.append(row)
    notes.column_dimensions["A"].width = 112
    notes["A1"].font = Font(name="Aptos Display", size=16, bold=True, color="183153")
    for idx in (4, 5, 6, 10, 11):
        notes[f"A{idx}"].font = Font(bold=True, color="9A3412")
    notes.sheet_view.showGridLines = False

    for db in ("lens", "overton"):
        passing, source_cols = load(db)
        # Source columns are appended, ensuring all the metrics in evaluator are retained.
        extra = [f"CSV: {c}" for c in source_cols]
        headers = SCOPUS_HEADERS + ADDITIONAL + extra
        converted = [data_record(row, db, source_cols) for row in passing]
        ws = wb.create_sheet(f"{db.title()} passing")
        style_sheet(ws, headers, converted, db)
        top = wb.create_sheet(f"{db.title()} top 20")
        style_sheet(top, headers, converted[:20], db)
        # Optional consistency check; warn but don't silently drop valid ranked rows.
        expected = {"lens": 884, "overton": 6}[db]
        if len(passing) != expected:
            print(f"WARNING: expected {expected} passing {db} configurations from previous run; got {len(passing)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)
    print(f"\nCREATED: {OUTPUT}")
    print("Sheets:", ", ".join(ws.title for ws in wb.worksheets))


if __name__ == "__main__":
    main()
