#!/usr/bin/env python3
"""
Export BERTopic results for Scopus, Lens and Overton.

Outputs:
1. Lens full grid CSV
2. Overton full grid CSV
3. Scopus 31-model Semantic Silhouette CSV
4. Scopus grid CSV, if an existing grid table is found
5. Consolidated Excel workbook

All original CSV columns are preserved.
No unverified join between Scopus model names and Grid_id.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("Results/bertopic_tuning")
OUT = ROOT / "ALL_BERTOPIC_RESULTS"
OUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "Outlier": ("Outlier_pts", 18),
    "Kappa": ("Label match_pts", 18),
    "NPMI": ("NPMI_pts", 14),
    "Diversity": ("Diversity_pts", 14),
    "Coverage": ("Label coverage_pts", 7),
    "Cohesion": ("Cohesion_pts", 7),
    "Size": ("Size_pts", 7),
    "Stability": ("Stability_pts", 7),
    "Semantic Silhouette": ("Semantic silhouette_pts", 8),
}

FRONT = [
    "Rank", "Grid_id", "Model",
    "n_neighbors", "n_components",
    "min_cluster_size_pct",
    "min_cluster_size_papers",
    "min_samples", "Selection",
    "Topics", "Collapsed",
    "Outlier_pct", "Largest_topic_pct",
    "Label_match_kappa",
    "NPMI", "Diversity_pct",
    "Label_coverage_pct",
    "Cohesion", "Stability_ARI",
    "Silhouette_minilm",
    "Weighted_score", "Mean_rank",
]


def arrange(d):
    first = [c for c in FRONT if c in d.columns]
    rest = [c for c in d.columns if c not in first]
    return d[first + rest]


def add_contributions(d):
    d = d.copy()

    for name, (pts, weight) in WEIGHTS.items():
        if pts in d.columns:
            d[name + "_contrib"] = (
                pd.to_numeric(
                    d[pts], errors="coerce"
                ) * weight / 3
            )

    return d


def read_lens_overton(db):
    p = (
        ROOT / db / "scopus_aligned_grid_v2"
        / "tables" / f"{db}_ranked.csv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    d = pd.read_csv(p)
    d = add_contributions(d)

    d["Database"] = db.upper()
    d["Evaluation_status"] = np.where(
        d["Weighted_score"].notna(),
        "FINAL_9_CRITERIA",
        np.where(
            d.get(
                "Eligible",
                pd.Series(False, index=d.index)
            ).astype(str).str.lower().isin(
                ["true", "1"]
            ),
            "ELIGIBLE_NOT_STABILITY_TESTED",
            "INELIGIBLE",
        ),
    )

    return arrange(d)


def read_scopus_silhouette():
    p = (
        ROOT / "scopus"
        / "harmonized_9criteria"
        / "scopus_semantic_silhouette.csv"
    )

    if not p.exists():
        raise FileNotFoundError(p)

    d = pd.read_csv(p)

    if d["Model"].duplicated().any():
        raise ValueError(
            "Duplicate Scopus model names"
        )

    d["Database"] = "SCOPUS"
    d["Evaluation_status"] = (
        "SAVED_MODEL_SILHOUETTE"
    )

    return arrange(d)


def find_scopus_grid():
    """
    Discover existing Scopus grid tables.
    Do not guess a join to named models.
    """

    base = ROOT / "scopus"

    candidates = []

    for p in base.rglob("*.csv"):
        if (
            "harmonized_9criteria" in p.parts
            or "document_topics.csv" == p.name
        ):
            continue

        name = p.name.lower()

        if any(
            word in name
            for word in [
                "ranked", "grid",
                "evaluation", "scores",
            ]
        ):
            candidates.append(p)

    valid = []

    for p in candidates:
        try:
            sample = pd.read_csv(p, nrows=3)

            if (
                "Grid_id" in sample.columns
                or (
                    "n_neighbors" in sample.columns
                    and "n_components" in sample.columns
                )
            ):
                valid.append(p)

        except Exception:
            continue

    if not valid:
        return None, None

    valid.sort(
        key=lambda p: (
            "ranked" in p.name.lower(),
            "final" in p.name.lower(),
            p.stat().st_size,
        ),
        reverse=True,
    )

    chosen = valid[0]
    d = pd.read_csv(chosen)

    d["Database"] = "SCOPUS"
    d["Source_file"] = str(chosen)

    return arrange(d), chosen


def write_csv(d, filename):
    p = OUT / filename
    d.to_csv(p, index=False)
    print(
        f"SAVED: {p} "
        f"({len(d)} rows, {len(d.columns)} columns)"
    )


def main():
    lens = read_lens_overton("lens")
    overton = read_lens_overton("overton")
    scopus = read_scopus_silhouette()

    scopus_grid, source = find_scopus_grid()

    write_csv(
        lens, "LENS_ALL_PARAMETERS.csv"
    )
    write_csv(
        overton, "OVERTON_ALL_PARAMETERS.csv"
    )
    write_csv(
        scopus, "SCOPUS_SILHOUETTE_ALL_MODELS.csv"
    )

    if scopus_grid is not None:
        write_csv(
            scopus_grid,
            "SCOPUS_GRID_ALL_PARAMETERS.csv"
        )
        print("Scopus grid source:", source)
    else:
        print(
            "WARNING: No Scopus grid CSV found. "
            "Saved-model results will still be exported."
        )

    complete = pd.concat(
        [lens, overton],
        ignore_index=True,
    )

    write_csv(
        complete,
        "LENS_OVERTON_COMBINED.csv"
    )

    xlsx = OUT / "ALL_BERTOPIC_RESULTS.xlsx"

    with pd.ExcelWriter(
        xlsx, engine="openpyxl"
    ) as writer:

        lens.to_excel(
            writer,
            sheet_name="Lens All",
            index=False,
        )

        overton.to_excel(
            writer,
            sheet_name="Overton All",
            index=False,
        )

        scopus.to_excel(
            writer,
            sheet_name="Scopus 31 Models",
            index=False,
        )

        if scopus_grid is not None:
            scopus_grid.to_excel(
                writer,
                sheet_name="Scopus Grid",
                index=False,
            )

        lens.loc[
            lens["Weighted_score"].notna()
        ].to_excel(
            writer,
            sheet_name="Lens Final Ranked",
            index=False,
        )

        overton.loc[
            overton["Weighted_score"].notna()
        ].to_excel(
            writer,
            sheet_name="Overton Final Ranked",
            index=False,
        )

        summary = pd.DataFrame([
            {
                "Database": "Lens",
                "Rows": len(lens),
                "Final_scored": int(
                    lens.Weighted_score.notna().sum()
                ),
                "Semantic_silhouette_values": int(
                    lens.Silhouette_minilm.notna().sum()
                ),
            },
            {
                "Database": "Overton",
                "Rows": len(overton),
                "Final_scored": int(
                    overton.Weighted_score.notna().sum()
                ),
                "Semantic_silhouette_values": int(
                    overton.Silhouette_minilm.notna().sum()
                ),
            },
            {
                "Database": "Scopus saved models",
                "Rows": len(scopus),
                "Final_scored": 0,
                "Semantic_silhouette_values": int(
                    scopus.Silhouette_minilm.notna().sum()
                ),
            },
        ])

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        for ws in writer.book.worksheets:
            ws.freeze_panes = "D2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.font = cell.font.copy(
                    bold=True,
                    color="FFFFFF",
                )
                cell.fill = cell.fill.copy(
                    fgColor="17365D",
                    fill_type="solid",
                )

            for col in ws.columns:
                letter = col[0].column_letter
                ws.column_dimensions[letter].width = 22

            ws.sheet_view.zoomScale = 85

    print("\nEXCEL CREATED:", xlsx)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
