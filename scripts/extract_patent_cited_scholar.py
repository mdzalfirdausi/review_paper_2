from pathlib import Path
import re

import pandas as pd


# =========================================================
# Project directories
# =========================================================

# project_root/
# ├── data/
# │   ├── lens/
# │   │   └── lens_patent_cited_dois.txt
# │   ├── scopus/
# │   │   └── scopus_full.xlsx
# │   └── lda/
# │
# └── scripts/
#     └── extract_patent_cited_scholar.py

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

LENS_DIR = (
    PROJECT_DIR
    / "data"
    / "lens"
)

SCOPUS_DIR = (
    PROJECT_DIR
    / "data"
    / "scopus"
)

LDA_DIR = (
    PROJECT_DIR
    / "data"
    / "lda"
)

LDA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Files
# =========================================================

DOI_FILE = (
    LENS_DIR
    / "lens_patent_cited_dois.txt"
)

SCOPUS_FILE = (
    SCOPUS_DIR
    / "scopus_full.xlsx"
)

OUTPUT_FILE = (
    LDA_DIR
    / "lens_from_scopus_scholar.xlsx"
)


# =========================================================
# Output columns
# =========================================================

OUTPUT_COLUMNS = [
    "Link",
    "Authors",
    "Author full names",
    "Author(s) ID",
    "Title",
    "Year",
    "Source title",
    "Cited by",
    "Affiliations",
    "Publisher",
    "Abbreviated Source Title",
    "DOI",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "Funding Details",
    "Funding Texts",
]


# =========================================================
# DOI normalization
# =========================================================

def normalize_doi(value):

    if pd.isna(value):
        return None

    doi = str(value).strip()

    if not doi:
        return None

    # Remove DOI: prefix if present.
    doi = re.sub(
        r"(?i)^doi:\s*",
        "",
        doi,
    )

    # Remove DOI URL if present.
    doi = re.sub(
        r"(?i)^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
    )

    doi = (
        doi
        .strip()
        .rstrip(".,;")
        .lower()
    )

    return doi or None


# =========================================================
# Validate input files
# =========================================================

for file in [
    DOI_FILE,
    SCOPUS_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            f"File not found: {file}"
        )


print("Project directory :", PROJECT_DIR)
print("DOI file          :", DOI_FILE)
print("Scopus file       :", SCOPUS_FILE)
print("Output file       :", OUTPUT_FILE)
print()


# =========================================================
# Read Lens patent-cited DOIs
# =========================================================

raw_dois = (
    DOI_FILE
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

lens_dois = [
    normalize_doi(value)
    for value in raw_dois
]

lens_dois = [
    doi
    for doi in lens_dois
    if doi is not None
]

lens_doi_set = set(
    lens_dois
)


print("Lens patent-cited DOI list")
print("--------------------------")

print(
    f"DOI lines     : "
    f"{len(raw_dois):,}"
)

print(
    f"Valid DOIs    : "
    f"{len(lens_dois):,}"
)

print(
    f"Unique DOIs   : "
    f"{len(lens_doi_set):,}"
)

print()


# =========================================================
# Read full Scopus dataset
# =========================================================

scopus = pd.read_excel(
    SCOPUS_FILE
)


print("Scopus full dataset")
print("-------------------")

print(
    f"Documents : "
    f"{len(scopus):,}"
)

print(
    f"Columns   : "
    f"{len(scopus.columns):,}"
)

print()


# =========================================================
# Validate required columns
# =========================================================

missing_columns = [
    column
    for column in OUTPUT_COLUMNS
    if column not in scopus.columns
]

if missing_columns:

    print("Available columns:")

    for column in scopus.columns:
        print(
            f"  - {column}"
        )

    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


# =========================================================
# Normalize Scopus DOI
# =========================================================

scopus = scopus.copy()

scopus["_doi_normalized"] = (
    scopus["DOI"]
    .apply(
        normalize_doi
    )
)


# =========================================================
# Select the Lens patent-cited publications
# =========================================================

patent_cited = (
    scopus[
        scopus[
            "_doi_normalized"
        ].isin(
            lens_doi_set
        )
    ]
    .copy()
)


# =========================================================
# Check matching
# =========================================================

matched_dois = set(
    patent_cited[
        "_doi_normalized"
    ]
    .dropna()
)

unmatched_dois = (
    lens_doi_set
    - matched_dois
)


print("DOI matching")
print("------------")

print(
    f"Requested DOIs : "
    f"{len(lens_doi_set):,}"
)

print(
    f"Matched DOIs   : "
    f"{len(matched_dois):,}"
)

print(
    f"Matched rows   : "
    f"{len(patent_cited):,}"
)

print(
    f"Unmatched DOIs : "
    f"{len(unmatched_dois):,}"
)

print()


# =========================================================
# Duplicate DOI check
# =========================================================

duplicate_mask = (
    patent_cited[
        "_doi_normalized"
    ]
    .duplicated(
        keep=False
    )
)

n_duplicate_rows = (
    duplicate_mask.sum()
)

print(
    f"Duplicate matched rows : "
    f"{n_duplicate_rows:,}"
)

print()


if n_duplicate_rows > 0:

    print("Duplicated matched publications")
    print("-------------------------------")

    print(
        patent_cited.loc[
            duplicate_mask,
            [
                "Title",
                "DOI",
            ],
        ]
        .sort_values(
            "DOI"
        )
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Keep one scholarly publication per DOI
# =========================================================

patent_cited = (
    patent_cited
    .drop_duplicates(
        subset="_doi_normalized",
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# Keep requested columns only
# =========================================================

patent_cited = (
    patent_cited[
        OUTPUT_COLUMNS
    ]
    .copy()
)


# =========================================================
# Save Excel
# =========================================================

patent_cited.to_excel(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# Final validation
# =========================================================

print("Final patent-cited scholarly corpus")
print("-----------------------------------")

print(
    f"Documents         : "
    f"{len(patent_cited):,}"
)

print(
    f"Columns           : "
    f"{len(patent_cited.columns):,}"
)

print(
    f"With abstract     : "
    f"{patent_cited['Abstract'].notna().sum():,}"
)

print(
    f"Missing abstract  : "
    f"{patent_cited['Abstract'].isna().sum():,}"
)

print(
    f"Unique DOI        : "
    f"{patent_cited['DOI'].nunique():,}"
)

print()

print(
    "Saved:"
)

print(
    OUTPUT_FILE
)

print()


# =========================================================
# Expected result check
# =========================================================

if (
    len(lens_doi_set) == 778
    and
    len(matched_dois) == 778
    and
    len(patent_cited) == 778
):

    print("CHECK PASSED")
    print("------------")

    print(
        "All 778 patent-cited scholarly "
        "publications were recovered from "
        "scopus_full.xlsx."
    )

elif len(matched_dois) == len(lens_doi_set):

    print("CHECK PASSED")
    print("------------")

    print(
        "All Lens DOIs were recovered from "
        "scopus_full.xlsx."
    )

else:

    print("CHECK WARNING")
    print("-------------")

    print(
        f"{len(unmatched_dois):,} DOI(s) "
        f"were not matched."
    )

    if unmatched_dois:

        print()
        print("Unmatched DOIs:")

        for doi in sorted(
            unmatched_dois
        ):

            print(
                doi
            )