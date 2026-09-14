# Comparative Topic Modeling of Policy-Facing and Academic Power-System Research

This repository contains the data-processing and topic-modeling workflow used to compare thematic emphasis in **policy-facing literature (Overton)** and **academic literature (Scopus)** related to power systems, machine learning, and optimization.

The analysis is designed around the research question:

> **What topics characterize power-system, machine-learning, and optimization research appearing in Overton versus Scopus, and how has their relative thematic emphasis evolved over time?**

The workflow combines Python and R. Python is used for data preparation, domain screening, diagnostics, post-processing, topic interpretation, cross-corpus comparison, and visualization. R is used for text preprocessing, document-term matrix construction, and Latent Dirichlet Allocation (LDA) estimation.

---

## Repository Workflow

The analysis is organized into the following main stages:

```text
Overton policy-document export
        |
        v
Extract and clean cited DOIs
        |
        v
Construct Scopus DOI queries
        |
        v
Overton + Scopus publication datasets
        |
        v
Data cleaning and deduplication
        |
        v
Power/energy domain relevance screening
        |
        v
Text preprocessing
        |
        v
Document-term matrix construction
        |
        v
Vocabulary filtering
        |
        v
LDA topic-number screening
        |
        v
Final LDA models
        |
        +------------------+
        |                  |
        v                  v
     Overton             Scopus
      K = 50              K = 50
        |                  |
        +--------+---------+
                 |
                 v
        Topic interpretation
                 |
                 v
      Common meta-theme mapping
                 |
                 v
       Cross-corpus prevalence
                 |
                 v
         Temporal comparison
```

---

## Repository Structure

A typical repository layout is:

```text
.
├── data/
│   ├── overton_policy_doc_*.csv
│   ├── overton_full_clean.xlsx
│   ├── scopus_full_clean.xlsx
│   ├── cleaned_dois.txt
│   ├── recovered_dois.txt
│   ├── rejected_values.txt
│   └── scopus_doi_queries.txt
│
├── notebooks/
│   ├── description.ipynb
│   ├── 1.data_prep.ipynb
│   └── 2.topic_modeling.ipynb
│
├── output/
│   ├── r_preprocessing/
│   ├── lda/
│   ├── domain_filtered_metadata/
│   ├── topic_interpretation/
│   ├── topic_prevalence/
│   ├── meta_theme_analysis/
│   ├── temporal_analysis/
│   └── figures/
│
├── 0.doi_filter.py
├── README.md
└── requirements.txt
```

The `data/` and `output/` directories may be excluded from Git using `.gitignore` when they contain large, licensed, downloaded, or generated files.

---

# Requirements

## Python

Python **3.10 or newer** is recommended.

The principal Python dependencies are:

```text
pandas
numpy
matplotlib
openpyxl
scipy
statsmodels
jupyter
```

`openpyxl` is required by pandas for reading the `.xlsx` input files.

Several additional modules used by the notebooks are part of the Python standard library and therefore do not need separate installation:

```text
pathlib
re
math
collections
subprocess
shutil
tempfile
```

### Python installation

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib openpyxl scipy statsmodels jupyter
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib openpyxl scipy statsmodels jupyter
```

Alternatively, using Conda:

```bash
conda create -n topic-modeling python=3.11
conda activate topic-modeling

conda install pandas numpy matplotlib openpyxl scipy statsmodels jupyter
```

---

# R Requirements

R must be installed separately because the Python notebook calls R scripts using:

```bash
Rscript
```

The workflow requires the following R packages:

```r
tm
SnowballC
slam
topicmodels
```

Install them from an R session with:

```r
install.packages(
    c(
        "tm",
        "SnowballC",
        "slam",
        "topicmodels"
    )
)
```

Verify that `Rscript` is available from the terminal:

```bash
Rscript --version
```

and verify the required packages with:

```bash
Rscript -e 'library(tm); library(SnowballC); library(slam); library(topicmodels); cat("R dependencies OK\n")'
```

If this prints:

```text
R dependencies OK
```

the R environment is ready.

## Important: Rscript path

The topic-modeling notebook launches R through Python's `subprocess` module.

The portable configuration is:

```python
import shutil
from pathlib import Path

rscript_path = shutil.which("Rscript")

if rscript_path is None:
    raise FileNotFoundError(
        "Rscript was not found in PATH."
    )

RSCRIPT = Path(rscript_path)
```

Do **not** commit a machine-specific path such as:

```text
/nfs/username/miniconda3/envs/.../bin/Rscript
```

to the final reusable notebook.

`Rscript` should instead be available through the user's `PATH`.

---

# 1. DOI Extraction and Scopus Retrieval

The DOI-processing stage starts from an Overton policy-document export containing a:

```text
Cited DOIs
```

column.

Run:

```bash
python 0.doi_filter.py
```

The script normalizes DOI strings, attempts to recover valid DOIs from contaminated citation metadata, removes duplicates, and writes:

```text
data/cleaned_dois.txt
data/recovered_dois.txt
data/rejected_values.txt
data/scopus_doi_queries.txt
```

`cleaned_dois.txt` contains one unique DOI per line.

The Scopus query file contains DOI queries in batches suitable for Scopus Advanced Search.

### Important

The DOI repair rules contain patterns identified from the source data. When applying the workflow to a new Overton export, the rejected and recovered DOI lists should be inspected manually.

Do not assume that all malformed DOI patterns occurring in a new dataset are covered by the existing rules.

---

# 2. Data Preparation

Run:

```text
notebooks/1.data_prep.ipynb
```

This stage prepares the cleaned Overton and Scopus publication datasets used by the topic-modeling workflow.

The expected principal outputs are:

```text
data/overton_full_clean.xlsx
data/scopus_full_clean.xlsx
```

The data-preparation stage should be reviewed whenever the source database schema changes.

In particular, verify:

- publication identifiers;
- DOI normalization;
- duplicate handling;
- missing abstracts;
- publication years;
- title and abstract columns.

---

# 3. Domain Relevance Screening

The original Overton and Scopus retrievals are not assumed to have identical substantive scope.

Before LDA, documents are screened using a common set of power- and energy-domain concept groups.

The screening considers terminology associated with areas such as:

- electricity and electrical grids;
- power systems;
- power-system operation and power flow;
- generation resources;
- transmission and distribution;
- renewable energy;
- storage and electric vehicles;
- power electronics;
- load and demand;
- energy systems.

The same screening logic is applied to both corpora.

For the current dataset, the final retained corpora contain:

| Corpus | Documents |
|---|---:|
| Overton | 5,045 |
| Scopus | 12,042 |

## Manual validation required

The domain-screening dictionary is **research-question specific**.

It should not be treated as a universal classifier.

When applying the workflow to another dataset:

1. inspect random retained documents;
2. inspect random rejected documents;
3. identify false positives and false negatives;
4. revise the domain vocabulary if necessary;
5. freeze the screening rule before final analysis.

Broad standalone terms such as `system`, `network`, `energy`, or `optimization` should be used cautiously because they may admit large amounts of unrelated literature.

---

# 4. Text Preprocessing

Text preprocessing is performed in R using:

```text
tm
SnowballC
```

The main preprocessing sequence is:

```text
lowercase
    ↓
remove punctuation
    ↓
remove numbers
    ↓
normalize whitespace
    ↓
remove English stopwords
    ↓
remove query-specific stopwords
    ↓
English stemming
```

For the current analysis, query-specific terms such as:

```text
power
flow
machine
learning
optimization
optimisation
```

are removed.

This prevents the LDA model from merely rediscovering terminology that was already imposed by the retrieval query.

## Important

Query-specific stopwords must be reconsidered whenever the search query changes.

---

# 5. Document-Term Matrix and Vocabulary Filtering

The DTM is constructed in R using `tm` and `slam`.

Terms shorter than three characters are excluded.

A proportional minimum document-frequency rule is used:

```text
minimum document frequency ≈ 0.2% of corpus documents
```

For the current corpora:

| Corpus | Documents | Minimum DF | Effective DF | Vocabulary |
|---|---:|---:|---:|---:|
| Overton | 5,045 | 11 | 0.218% | 2,865 |
| Scopus | 12,042 | 25 | 0.208% | 2,737 |

The proportional rule is used because applying the same absolute document-frequency threshold to corpora of substantially different sizes would impose different effective vocabulary filters.

## Important

`0.2%` is an analysis configuration, not a universal value.

For a new corpus, vocabulary-size diagnostics should be inspected before selecting the final threshold.

---

# 6. LDA Topic Modeling

LDA is estimated using the R package:

```r
topicmodels
```

with Gibbs sampling.

## Training and held-out split

For model selection, each corpus uses an 80/20 training/held-out split with:

```text
random seed = 123
```

The current split is:

| Corpus | Training | Held-out |
|---|---:|---:|
| Overton | 4,036 | 1,009 |
| Scopus | 9,633 | 2,409 |

---

## Topic-number screening

A coarse search evaluates:

```text
K = 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100
```

using relatively short Gibbs chains.

Candidate models are then evaluated more carefully at:

```text
K = 30, 50, 70, 100
```

using:

- held-out perplexity;
- semantic coherence;
- inter-topic similarity;
- topic interpretability.

The final working thematic resolution for the present analysis is:

```text
Overton: K = 50
Scopus : K = 50
```

## Important

`K = 50` must **not** be treated as a universal setting.

A different dataset should repeat the topic-number screening and interpretation procedure.

Lower perplexity alone should not determine the final number of topics because increasing `K` can improve predictive fit while simultaneously producing less coherent or excessively fragmented topics.

---

# 7. Final LDA Estimation

After topic-number selection, the final models are refitted using all domain-relevant documents.

Current final configuration:

```text
K          = 50
seed       = 123
burn-in    = 2,000
iterations = 10,000
thin       = 100
```

The final models export files including:

```text
beta.csv
theta.csv
top_terms.csv
document_topics.csv
topic_prevalence.csv
model_summary.csv
```

where:

- **beta** contains topic-word probabilities;
- **theta** contains document-topic probabilities.

Each document can belong probabilistically to multiple topics.

---

# 8. Topic Interpretation

Topic interpretation is **not fully automated**.

For each latent topic, the workflow extracts:

- highest-probability terms;
- five publications with the highest topic probability;
- publication titles;
- publication metadata.

Human-readable topic labels are then assigned by jointly inspecting the top terms and representative publications.

## Important

Do not label topics using top words alone.

Some LDA topics are highly coherent, while others may be generic, cross-cutting, or noisy. Ambiguous topics should receive appropriately broad labels rather than being forced into a narrowly defined technical category.

When using a different dataset, **all topic labels must be reassessed manually**.

---

# 9. Cross-Corpus Meta-Themes

Because Overton and Scopus are modeled independently:

```text
Overton Topic 1 != Scopus Topic 1
```

Topic identifiers have no direct meaning across separately estimated LDA models.

The 50 topics from each corpus are therefore mapped to a common set of higher-level meta-themes.

The current analysis uses 11 meta-themes:

1. Optimization and Computational Methods
2. Machine Learning and Data-Driven Methods
3. Power-System Operation, Stability, and Reliability
4. Distribution Systems, Microgrids, and Smart Grids
5. Renewable Energy and Resource Integration
6. Energy Storage and Electric Vehicles
7. Markets, Flexibility, and Demand Response
8. Power Electronics and Grid Technologies
9. Climate, Environment, and Decarbonization
10. Policy, Society, and Energy Transition
11. Cross-Cutting and Review Studies

## Manual interpretation required

This topic-to-meta-theme mapping is a substantive coding layer.

The code can automatically aggregate probabilities **after the mapping has been defined**, but the mapping itself should be inspected and justified by the researcher.

For a different research question or substantially different corpus, the meta-theme taxonomy may need to change.

---

# 10. Topic and Meta-Theme Prevalence

Topic prevalence is probability weighted.

For topic \(k\):

\[
P_k =
\frac{1}{N}
\sum_{d=1}^{N}
\theta_{dk}.
\]

This uses the complete LDA mixed-membership distribution rather than counting only each document's dominant topic.

Topic probabilities are then aggregated into the common meta-themes.

The Overton–Scopus prevalence difference is reported in **percentage points**:

\[
\Delta_m =
P_{m,\mathrm{Overton}}
-
P_{m,\mathrm{Scopus}}.
\]

Therefore:

```text
positive difference -> relatively greater Overton representation
negative difference -> relatively greater Scopus representation
```

These results should be interpreted as differences in the **relative thematic representation of the two corpora**, not as direct measurements of individual policymakers' or academics' preferences.

---

# 11. Temporal Analysis

Annual meta-theme prevalence is calculated from the complete document-level topic probability distributions.

The current comparative temporal window is:

```text
2004–2025
```

This interval was selected after examining annual publication counts.

Within this interval:

```text
22 years are represented in both corpora
no years are missing
each corpus contains at least 30 documents per year
```

The temporal window should **not** automatically be reused for another dataset.

For new data, annual publication coverage should be inspected first.

Potentially incomplete future/current publication years should also be handled carefully.

---

# 12. Figures and Outputs

Publication-quality figures are saved as PDF under:

```text
output/figures/
```

PDF is used to preserve vector graphics for manuscript preparation.

Generated figures include, among others:

```text
meta_theme_prevalence_overton_vs_scopus.pdf
meta_theme_prevalence_gap.pdf

overton_meta_theme_temporal_heatmap_2004_2025.pdf
scopus_meta_theme_temporal_heatmap_2004_2025.pdf

meta_theme_temporal_trend_comparison.pdf
```

Generated numerical results are stored separately under:

```text
output/topic_interpretation/
output/topic_prevalence/
output/meta_theme_analysis/
output/temporal_analysis/
```

---

# What Is Automated and What Requires Manual Review?

The workflow intentionally separates reproducible computation from substantive interpretation.

| Stage | Automated | Manual review required |
|---|:---:|:---:|
| DOI normalization | ✓ | ✓ malformed/recovered values |
| Deduplication | ✓ | ✓ ambiguous identifiers |
| Domain screening | ✓ after definition | **✓ essential** |
| Text preprocessing | ✓ | configuration |
| DTM construction | ✓ | threshold selection |
| LDA fitting | ✓ | |
| Perplexity/coherence diagnostics | ✓ | |
| Topic-number selection | partly | **✓ essential** |
| Representative-document ranking | ✓ | |
| Topic labeling | | **✓ essential** |
| Topic prevalence | ✓ | |
| Meta-theme mapping | | **✓ essential** |
| Meta-theme aggregation | ✓ | |
| Temporal aggregation | ✓ | temporal-window selection |
| Figures | ✓ | interpretation |

A useful principle for reproducing or extending the study is:

> **Automate the calculation; manually validate the interpretation.**

---

# Using the Workflow with a Different Dataset

Do **not** simply replace the input files and run the complete notebook unchanged.

At minimum, reconsider:

1. **Input schema**  
   Verify column names, DOI fields, title, abstract, and publication year.

2. **Deduplication rules**  
   DOI behavior and publication identifiers may differ between databases.

3. **Domain-screening vocabulary**  
   The current dictionary was developed for power/energy research.

4. **Query-specific stopwords**  
   These must reflect the retrieval query used for the new dataset.

5. **Vocabulary filtering**  
   Re-evaluate the minimum document-frequency threshold.

6. **Number of LDA topics (`K`)**  
   Repeat model selection. Do not automatically reuse `K = 50`.

7. **Topic interpretation**  
   Inspect top terms and representative publications again.

8. **Meta-theme taxonomy**  
   The current 11 categories may not be appropriate for another domain.

9. **Topic-to-meta-theme mapping**  
   This must be manually reassessed.

10. **Temporal window**  
    Determine the period from actual annual document coverage.

---

# Reproducibility

Random seeds are fixed where appropriate.

The current LDA workflow uses:

```text
seed = 123
```

Intermediate outputs are saved to disk so expensive preprocessing and model-fitting stages do not need to be repeated unnecessarily.

For exact reproduction, record:

- Python version;
- R version;
- Python package versions;
- R package versions;
- source-data export dates;
- final screening dictionary;
- final topic labels;
- final topic-to-meta-theme mappings.

Python package versions can be recorded with:

```bash
python -m pip freeze > requirements-lock.txt
```

R package information can be recorded with:

```bash
Rscript -e 'sessionInfo()'
```

For a more reproducible R environment, consider using `renv`.

---

# Recommended Execution Order

Run the workflow in the following order:

```text
1. 0.doi_filter.py

2. Retrieve/export the corresponding Scopus records

3. notebooks/1.data_prep.ipynb

4. notebooks/2.topic_modeling.ipynb
```

The topic-modeling notebook performs the remaining domain-screening, preprocessing, LDA model-selection, final-model, interpretation, prevalence, cross-corpus, and temporal-analysis stages.

---

# Data Availability

Raw Overton and Scopus exports may be subject to database licensing or redistribution restrictions.

For that reason, raw source datasets should not automatically be committed to a public repository.

Where redistribution is not permitted, the repository should provide:

- processing code;
- expected input schemas;
- configuration;
- derived non-restricted outputs where permitted;
- instructions for users to obtain/export the source data independently.

---

# Citation

If you use this repository or workflow, please cite the associated paper:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

The citation information should be updated after publication.

---

# License

Add the repository's software license here.

For example:

```text
MIT License
```

Note that a software license does not override the licensing terms of Overton, Scopus, or other source datasets.