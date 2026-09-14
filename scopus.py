from pathlib import Path
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

SCOPUS_CLEAN_FILE = (
    Path("data")
    / "scopus_full_clean.xlsx"
)

OUTPUT_FILE = (
    Path("data")
    / "scopus_cleaned_dois.txt"
)

# ---------------------------------------------------------
# Load Scopus data
# ---------------------------------------------------------

scopus = pd.read_excel(
    SCOPUS_CLEAN_FILE
)

print(
    f"Scopus records : {len(scopus):,}"
)

# ---------------------------------------------------------
# Extract normalized DOIs
# ---------------------------------------------------------

dois = (
    scopus["DOI_normalized"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
)

# Remove blank / invalid placeholder values.
dois = dois[
    ~dois.isin([
        "",
        "0",
        "nan",
        "none",
    ])
]

# Remove duplicate DOIs while preserving first occurrence.
dois = dois.drop_duplicates()

# ---------------------------------------------------------
# Save one DOI per line
# ---------------------------------------------------------

OUTPUT_FILE.write_text(
    "\n".join(dois),
    encoding="utf-8",
)

print(
    f"Unique valid DOIs : {len(dois):,}"
)

print(
    "Saved to:",
    OUTPUT_FILE
)