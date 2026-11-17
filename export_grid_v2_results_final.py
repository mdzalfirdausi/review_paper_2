#!/usr/bin/env python3
"""Export Lens and Overton final BERTopic results to Excel."""

from pathlib import Path
import csv
import json
import math
import re

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "Results" / "bertopic_tuning"
OUTPUT = BASE / "GRID_V2_RESULTS_LENS_OVERTON_FINAL.xlsx"

WEIGHTS = {
    "Outlier": 18,
    "Kappa": 18,
    "NPMI": 14,
    "Diversity": 14,
    "Coverage": 7,
    "Cohesion": 7,
    "Size": 7,
    "Stability": 7,
    "Semantic Silhouette": 8,
}

METRICS = [
    ("Outlier", "Outlier_pct", "Outlier_pts"),
    ("Kappa", "Label_match_kappa", "Label match_pts"),
    ("NPMI", "NPMI", "NPMI_pts"),
    ("Diversity", "Diversity_pct", "Diversity_pts"),
    ("Coverage", "Label_coverage_pct", "Label coverage_pts"),
    ("Cohesion", "Cohesion", "Cohesion_pts"),
    ("Size", "Largest_topic_pct", "Size_pts"),
    ("Stability", "Stability_ARI", "Stability_pts"),
    (
        "Semantic Silhouette",
        "Silhouette_minilm",
        "Semantic silhouette_pts",
    ),
]

PAT = re.compile(
    r"^(mpnet|biobert_sts|minilm)_nn(\d+)_nc(\d+)"
    r"_mcs([\d.]+)_ms(\d+)_(eom|leaf)$"
)

DISPLAY = {
    "mpnet": "paraphrase-mpnet-base-v2",
    "biobert_sts": "BioBERT-STS",
    "minilm": "all-MiniLM-L6-v2",
}


def num(x):
    if x is None or str(x).strip().lower() in (
        "", "nan", "none", "null"
    ):
        return None
    try:
        f = float(x)
        if not math.isfinite(f):
            return None
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def read(db):
    path = (
        BASE / db / "scopus_aligned_grid_v2"
        / "tables" / f"{db}_ranked.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)

    if len(rows) != 7776:
        raise ValueError(
            f"{db}: expected 7776 rows, found {len(rows)}"
        )
    return rows, cols


def seed_topics(db, grid_id):
    root = (
        BASE / db / "scopus_aligned_grid_v2" / "stability"
    )
    result = []

    for seed in (0, 1, 2):
        path = (
            root / f"{grid_id}_seed{seed}" / "summary.json"
        )
        if not path.exists():
            result.append(None)
            continue

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        result.append(num(data.get("Topics")))

    return result


def headers(raw):
    cols = [
        "Rank",
        "Evaluation status",
        "Eligible",
        "Grid_id",
        "Model",
        "n_neighbors",
        "n_components",
        "min_cluster_size_pct",
        "min_cluster_size_papers",
        "min_samples",
        "Selection",
        "Topics",
        "Collapsed",
        "Outlier %",
        "Largest topic %",
    ]

    for label, _, _ in METRICS:
        cols.extend([
            label + " value",
            label + " pts",
            label + " contrib",
        ])

    cols.extend([
        "Seed topics 0",
        "Seed topics 1",
        "Seed topics 2",
        "Seed topics (0/1/2)",
        "Stability ARI minimum",
        "Score",
        "Mean rank (tie-break)",
        "Total points",
        "G1",
        "G2",
        "G3",
    ])

    cols.extend("CSV: " + c for c in raw)
    return cols


def transform(row, db, raw):
    grid_id = row["Grid_id"]
    match = PAT.fullmatch(grid_id)

    if not match:
        raise ValueError(f"Unknown Grid_id: {grid_id}")

    model, nn, nc, mcs, ms, selection = match.groups()

    eligible = str(row.get("Eligible", "")).lower() in (
        "true", "1"
    )

    score = num(row.get("Weighted_score"))
    stability = num(row.get("Stability_ARI"))

    final = score is not None and stability is not None

    if final:
        status = "FINAL (9 criteria)"
    elif eligible:
        status = "ELIGIBLE - stability not tested"
    else:
        status = "INELIGIBLE"

    d = {
        "Rank": num(row.get("Rank")),
        "Evaluation status": status,
        "Eligible": eligible,
        "Grid_id": grid_id,
        "Model": DISPLAY[model],
        "n_neighbors": int(nn),
        "n_components": int(nc),
        "min_cluster_size_pct": float(mcs),
        "min_cluster_size_papers": num(
            row.get("min_cluster_size_papers")
        ),
        "min_samples": int(ms),
        "Selection": selection,
        "Topics": num(row.get("Topics")),
        "Collapsed": str(
            row.get("Collapsed", "")
        ).lower() in ("true", "1"),
        "Outlier %": num(row.get("Outlier_pct")),
        "Largest topic %": num(
            row.get("Largest_topic_pct")
        ),
    }

    for label, value_col, point_col in METRICS:
        value = num(row.get(value_col))
        points = num(row.get(point_col))

        d[label + " value"] = value
        d[label + " pts"] = points
        d[label + " contrib"] = (
            round(WEIGHTS[label] * points / 3, 10)
            if points is not None else None
        )

    seeds = (
        seed_topics(db, grid_id)
        if final else [None, None, None]
    )

    for i, value in enumerate(seeds):
        d[f"Seed topics {i}"] = value

    d["Seed topics (0/1/2)"] = (
        " / ".join(map(str, seeds))
        if all(x is not None for x in seeds)
        else None
    )

    d["Stability ARI minimum"] = num(
        row.get("Stability_ARI_min")
    )
    d["Score"] = score
    d["Mean rank (tie-break)"] = num(
        row.get("Mean_rank")
    )
    d["Total points"] = num(
        row.get("Total_points")
    )

    for name in ("G1", "G2", "G3"):
        d[name] = num(row.get(name))

    for c in raw:
        value = row.get(c)
        numeric = num(value)
        d["CSV: " + c] = (
            numeric if numeric is not None
            else value if value not in ("", "nan", "None")
            else None
        )

    if final:
        calculated = sum(
            d[label + " contrib"]
            for label in WEIGHTS
        )

        if abs(calculated - score) > 1e-5:
            raise ValueError(
                f"Score mismatch: {db}/{grid_id}: "
                f"{calculated} vs {score}"
            )

    return d


def make_sheet(wb, name, records, cols):
    ws = wb.create_sheet(name)
    ws.append(cols)

    for record in records:
        ws.append([record.get(c) for c in cols])

    ws.freeze_panes = "E2"
    ws.auto_filter.ref = (
        f"A1:{get_column_letter(len(cols))}"
        f"{len(records) + 1}"
    )
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 80

    for cell in ws[1]:
        cell.fill = PatternFill(
            "solid", fgColor="183153"
        )
        cell.font = Font(
            bold=True, color="FFFFFF", size=10
        )
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    ws.row_dimensions[1].height = 42

    for j, name in enumerate(cols, 1):
        width = (
            39 if name == "Grid_id"
            else 30 if name.startswith("CSV:")
            else 24
        )
        ws.column_dimensions[
            get_column_letter(j)
        ].width = width

    for cells in ws.iter_rows(min_row=2):
        if cells[0].row % 2 == 0:
            for cell in cells:
                cell.fill = PatternFill(
                    "solid", fgColor="F1F5F9"
                )

        for cell in cells:
            if isinstance(cell.value, float):
                cell.number_format = "0.000000"


def main():
    wb = Workbook()
    notes = wb.active
    notes.title = "README"

    lines = [
        "LENS / OVERTON FINAL BERTopic EVALUATION",
        "All 7,776 configurations per database are retained.",
        "Eligibility: 5-30 topics, outlier <10%, largest topic <50%.",
        "Lens: 884 eligible; 60 stability-tested.",
        "Overton: 6 eligible; all 6 stability-tested.",
        "Final score requires all nine criteria.",
        "Contribution = points / 3 * weight.",
        "Stability is mean pairwise ARI across seeds 0,1,2.",
        "Semantic Silhouette uses Silhouette_minilm.",
        "G1/G2/G3 are not inferred when absent.",
        "Scores are ranked within each corpus.",
        "Unmeasured metrics remain blank, not zero.",
    ]

    for line in lines:
        notes.append([line])

    notes.column_dimensions["A"].width = 110
    notes["A1"].font = Font(
        size=16, bold=True, color="183153"
    )

    for db in ("lens", "overton"):
        rows, raw = read(db)
        data = [transform(r, db, raw) for r in rows]

        eligible = [
            r for r in data if r["Eligible"]
        ]
        final = [
            r for r in data
            if r["Evaluation status"] == "FINAL (9 criteria)"
        ]

        expected = 60 if db == "lens" else 6

        if len(final) != expected:
            raise ValueError(
                f"{db}: expected {expected} final rows, "
                f"found {len(final)}"
            )

        cols = headers(raw)

        make_sheet(
            wb, db.title() + " ALL 7776",
            data, cols
        )
        make_sheet(
            wb, db.title() + " Eligible",
            eligible, cols
        )
        make_sheet(
            wb, db.title() + " Final Ranked",
            final, cols
        )
        make_sheet(
            wb, db.title() + " Top 20",
            final[:20], cols
        )

        print(
            f"{db.upper()}: "
            f"{len(data)} total, "
            f"{len(eligible)} eligible, "
            f"{len(final)} final"
        )

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )
    wb.save(OUTPUT)
    print("CREATED:", OUTPUT)


if __name__ == "__main__":
    main()
