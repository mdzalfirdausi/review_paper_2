from pathlib import Path
from urllib.parse import unquote
import argparse
import re

import pandas as pd
from openpyxl.styles import PatternFill, Font, Alignment


DATABASES = ("scopus", "lens", "overton")
PREFIX = {"scopus": "S", "lens": "L", "overton": "O"}


def normalize_doi(value):
    if pd.isna(value):
        return ""

    text = unquote(str(value)).strip()
    text = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)",
        "",
        text,
        flags=re.I,
    ).strip()

    # Avoid silently merging malformed or multiple DOI values.
    if re.fullmatch(r"10\.\d{4,9}/\S+", text, flags=re.I):
        return text.lower()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("representative_papers.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/topic_references"),
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype={"DOI": str},
    )
    df.columns = df.columns.str.strip()

    required = [
        "Database", "Topic", "Rank", "Title", "DOI",
        "Scopus_citations", "Patent_citations", "Policy_citations",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Database"] = df["Database"].astype(str).str.strip().str.lower()
    unexpected = set(df["Database"]) - set(DATABASES)
    if unexpected:
        raise ValueError(f"Unknown database values: {unexpected}")

    for column in ["Topic", "Rank"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if (
            df[column].isna().any()
            or (df[column] < 1).any()
            or (df[column] % 1 != 0).any()
        ):
            raise ValueError(f"{column} must contain positive integers.")
        df[column] = df[column].astype(int)

    if df.duplicated(["Database", "Topic", "Rank"]).any():
        raise ValueError("Duplicate ranks within a database/topic group.")

    for column in [
        "Scopus_citations", "Patent_citations", "Policy_citations"
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Topic_code"] = [
        PREFIX[database] + str(topic)
        for database, topic in zip(df["Database"], df["Topic"])
    ]
    df["doi_norm"] = df["DOI"].map(normalize_doi)

    # Keep the four ranked representatives already identified in the input.
    selected = df.loc[df["Rank"].between(1, 4)].copy()

    if selected.empty:
        raise ValueError("No rows with Rank 1–4 were found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Retain unresolved DOI rows for review; never merge by title alone.
    unresolved = selected.loc[selected["doi_norm"].eq("")].copy()
    valid = selected.loc[selected["doi_norm"].ne("")].copy()

    records = []
    for doi, group in valid.groupby("doi_norm", sort=False):
        present = set(group["Database"])
        count = len(present)

        if count == 3:
            overlap = "ALL THREE"
        elif count == 2:
            overlap = "TWO: " + " + ".join(
                database.title()
                for database in DATABASES
                if database in present
            )
        else:
            overlap = "ONE"

        # Prefer the Scopus title when that representative entry exists.
        preferred = group.assign(
            database_order=group["Database"].map(
                {database: i for i, database in enumerate(DATABASES)}
            )
        ).sort_values(["database_order", "Rank"])

        titles = preferred["Title"].dropna().astype(str).str.strip()
        titles = titles.loc[titles.ne("")]

        record = {
            "doi_norm": doi,
            "Title": titles.iloc[0] if len(titles) else "",
            "DOI_URL": f"https://doi.org/{doi}",
            "Database_count": count,
            "Overlap": overlap,
            "Representative_entries": len(group),
            "All_topics": "; ".join(
                sorted(
                    set(group["Topic_code"]),
                    key=lambda code: (
                        list(PREFIX.values()).index(code[0]),
                        int(code[1:]),
                    ),
                )
            ),
        }

        for database in DATABASES:
            subset = group.loc[group["Database"].eq(database)]
            record[f"In_{database}_representatives"] = not subset.empty
            record[f"{database.title()}_topics"] = "; ".join(
                sorted(
                    set(subset["Topic_code"]),
                    key=lambda code: int(code[1:]),
                )
            )

        # Preserve supplied values rather than summing repeated citations.
        for column in [
            "Scopus_citations", "Patent_citations", "Policy_citations"
        ]:
            values = sorted(group[column].dropna().unique())
            record[column] = "; ".join(f"{value:g}" for value in values)
            record[f"{column}_conflict"] = len(values) > 1

        records.append(record)

    unique = pd.DataFrame(records)
    if unique.empty:
        raise ValueError("No valid DOIs found; inspect the input DOI column.")

    unique = unique.sort_values(
        ["Database_count", "Representative_entries", "Title"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    all_three = unique.loc[unique["Database_count"].eq(3)].copy()
    two_or_more = unique.loc[unique["Database_count"].ge(2)].copy()

    # Add overlap information to the topic-by-topic reference list.
    selected = selected.merge(
        unique[["doi_norm", "Database_count", "Overlap"]],
        on="doi_norm",
        how="left",
        validate="many_to_one",
    )
    selected["Overlap"] = selected["Overlap"].fillna("DOI NEEDS REVIEW")
    selected["database_order"] = selected["Database"].map(
        {database: i for i, database in enumerate(DATABASES)}
    )
    selected = selected.sort_values(
        ["database_order", "Topic", "Rank"]
    ).drop(columns="database_order")

    topic_counts = (
        selected.groupby(["Database", "Topic_code"], sort=False)
        .size()
        .reset_index(name="Representative_count")
    )
    incomplete = topic_counts.loc[
        topic_counts["Representative_count"].ne(4)
    ]

    notes = pd.DataFrame({
        "Notes": [
            "Rows with Rank 1–4 are taken from representative_papers.csv.",
            "Rank is inherited from the source; this script does not "
            "establish rankings from full topic memberships.",
            "Overlap means presence in the supplied representative lists, "
            "not confirmed membership across the complete corpora.",
            "Repeated publications are combined by normalized DOI.",
            "Citation counts are not summed across repeated entries.",
            "Multiple citation values are retained and flagged as conflicts.",
            "Gold highlights ALL THREE; blue highlights TWO collections.",
        ]
    })

    sheets = {
        "By_topic": selected,
        "Unique_references": unique,
        "ALL_THREE": all_three,
        "TWO_OR_MORE": two_or_more,
        "DOI_review": unresolved,
        "Topic_counts": topic_counts,
        "Notes": notes,
    }

    for name, data in sheets.items():
        data.to_csv(
            args.out_dir / f"{name.lower()}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    workbook = args.out_dir / "opf_topic_references.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, data in sheets.items():
            data.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="203D54")
                cell.font = Font(color="FFFFFF", bold=True)

            headers = {
                cell.value: cell.column for cell in ws[1]
            }
            count_column = headers.get("Database_count")

            for row in ws.iter_rows(min_row=2):
                count = (
                    row[count_column - 1].value
                    if count_column else None
                )
                color = (
                    "FFE29A" if count == 3
                    else "DAEAF7" if count == 2
                    else None
                )
                for cell in row:
                    cell.alignment = Alignment(
                        vertical="top", wrap_text=True
                    )
                    if color:
                        cell.fill = PatternFill("solid", fgColor=color)
                    if count == 3:
                        cell.font = Font(bold=True)

            for column in ws.columns:
                header = column[0].value
                width = (
                    75 if header == "Title"
                    else 55 if header in {"DOI", "doi_norm", "DOI_URL"}
                    else 85 if header == "Notes"
                    else 25
                )
                ws.column_dimensions[column[0].column_letter].width = width

    (args.out_dir / "all_unique_dois.txt").write_text(
        "\n".join(unique["doi_norm"]) + "\n",
        encoding="utf-8",
    )

    print(f"Topic-reference entries: {len(selected)}")
    print(f"Unique valid DOIs: {len(unique)}")
    print(f"In two or more representative lists: {len(two_or_more)}")
    print(f"In all three representative lists: {len(all_three)}")
    print(f"DOIs needing review: {len(unresolved)}")
    if not incomplete.empty:
        print("\nTopics without exactly four representative entries:")
        print(incomplete.to_string(index=False))
    print(f"\nSaved: {workbook.resolve()}")


if __name__ == "__main__":
    main()