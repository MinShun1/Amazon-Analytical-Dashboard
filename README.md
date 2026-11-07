# E-Commerce Analytics Dashboard

An interactive Streamlit dashboard that turns an exploratory Jupyter notebook into a
structured business-analytics application for an Indian apparel e-commerce business.
It combines **descriptive analytics**, **statistical hypothesis testing** and the
**integration of existing machine-learning experiments** across five business domains.

> Machine learning is a supporting component. The goal is to show how existing models fit
> into an analytics workflow, not to maximise model performance. Model results are reported
> honestly, including weak ones.

<!-- Add a screenshot after your first run, e.g. docs/overview.png -->
<!-- ![Business Overview](docs/overview.png) -->

---

## Contents

- [Pages](#pages)
- [Repository structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Getting the data](#getting-the-data)
- [Building the artifacts](#building-the-artifacts)
- [Running the dashboard](#running-the-dashboard)
- [Running the tests](#running-the-tests)
- [Reproducibility](#reproducibility)
- [Methodology notes](#methodology-notes)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Git workflow](#git-workflow)

---

## Pages

| Page | Descriptive | Statistical | Predictive / models |
|---|---|---|---|
| **1 · Business Overview** | Cross-domain KPIs, monthly trends, highlights | Summary of all four tests | Summary of all five experiments |
| **2 · Amazon Analytics** | Orders, amount, status, categories, fulfilment, states, sizes (filterable) | Fulfilment × cancellation (χ², Cramér's V); amount across categories (Kruskal-Wallis + Dunn); feature screening | Cancellation prediction: Experiment 1 (10 models, June test) and Experiment 1.1 (6 models, May validation) with metrics, confusion matrices, reports, threshold explorer, prediction table |
| **3 · International Analytics** | Monthly sales, customer concentration (Pareto), sizes, SKU completeness, de-duplication audit (filterable) | Purchase frequency vs rate (Spearman); rate across size groups (Kruskal-Wallis + Dunn) | Customer segmentation (K-Means, k = 2); monthly sales forecasting (Naive, MA, SES, Holt) |
| **4 · Product Analytics** | Recorded stock by category, size, colour, design; data-quality panel (filterable) | — | Product price segmentation (K-Means, k = 5) |
| **5 · Financial & Operational** | Receipts, expense reconciliation, warehouse tariffs, P&L 2021 pricing reference | — | — |

Sidebar filters apply to the **Descriptive** tabs only. Statistical and model results are
fixed outputs of the notebook's analysis and are labelled as such.

---

## Repository structure

```text
Analytical Dashboard/
├── app.py                      # Streamlit entry point (st.navigation)
├── pages/                      # One file per dashboard page (layout only)
├── src/
│   ├── config.py               # Relative paths, constants, seeds, split dates
│   ├── data/
│   │   ├── loader.py           # Reads the 7 CSVs straight from the zip
│   │   ├── preprocessing.py    # Reproduces the notebook's effective cleaning
│   │   ├── validation.py       # Checks against the notebook's saved outputs
│   │   └── app_cache.py        # Streamlit-cached access to data and artifacts
│   ├── analysis/
│   │   ├── descriptive.py      # KPIs and aggregations
│   │   └── statistical.py      # Hypothesis tests and effect sizes
│   ├── models/
│   │   ├── cancellation.py     # Experiments 1 and 1.1
│   │   ├── segmentation.py     # International customer K-Means
│   │   ├── forecasting.py      # International monthly forecasting
│   │   ├── clustering.py       # Product price K-Means
│   │   └── artifacts.py        # Save/load results and models
│   ├── visualization/
│   │   ├── charts.py           # Plotly chart builders (shared style)
│   │   └── components.py       # KPI cards, method cards, filters, guards
│   └── utils/helpers.py        # Number/currency/date formatting
├── scripts/build_artifacts.py  # One-off build + reconciliation report
├── tests/                      # pytest: pipeline, statistics, headless page tests
│   └── reference_values.json   # Values taken from the notebook's saved outputs
├── notebooks/amazon-sales-report.ipynb   # Source notebook (fully executed)
├── Data/
│   ├── 1_Raw/                  # Put the Kaggle zip here (git-ignored)
│   └── 2_Processed/            # Generated parquet files (git-ignored)
├── artifacts/                  # Generated results and models (git-ignored)
├── requirements.txt            # Pinned runtime dependencies
├── requirements-dev.txt        # + pytest
└── .python-version             # 3.12
```

---

## Prerequisites

- **Python 3.12** (the notebook ran on Python 3.12; other versions may give slightly different model results)
- Git
- About 600 MB of free disk space (virtual environment, data and model artifacts)

Check your Python versions:

```bash
py -0p
```

(macOS/Linux: `python3.12 --version`)

---

## Setup

All commands run from the project root (`Analytical Dashboard/`).

### 1. Create the virtual environment

Windows (PowerShell):

```bash
py -3.12 -m venv .venv
```

macOS / Linux:

```bash
python3.12 -m venv .venv
```

### 2. Activate it

Windows (PowerShell):

```bash
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, allow local scripts for your user once, then activate again:

```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Windows (Git Bash): `source .venv/Scripts/activate` · macOS / Linux: `source .venv/bin/activate`

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements-dev.txt
```

Use `requirements.txt` instead if you don't need the tests.

### Managing the environment

- **Deactivate:** `deactivate`
- **Recreate after a requirements change:** delete `.venv/` and repeat steps 1–3.
- **Add a dependency:** `pip install <package>`, then add it with its exact version
  (`pip show <package>`) to `requirements.txt`, rebuild the artifacts and run the tests.
- **Do not upgrade pinned packages casually.** In particular, scikit-learn 1.6.1 is pinned
  because later versions change how `RandomForestClassifier(class_weight="balanced")`
  behaves, which changes the Random Forest results (see [Reproducibility](#reproducibility)).

---

## Getting the data

The raw data is **not** included in the repository.

1. Download the dataset from Kaggle:
   [Unlock Profits with E-Commerce Sales Data](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)
2. Save the downloaded archive as:

   ```text
   Data/1_Raw/E-Commerce Sales Dataset.zip
   ```

3. The zip must contain these seven files (at its root or in a sub-folder):

   | File | Dataset |
   |---|---|
   | `Amazon Sale Report.csv` | `amazon` |
   | `International sale Report.csv` | `international` |
   | `May-2022.csv` | `may_2022` |
   | `P  L March 2021.csv` (two spaces) | `pnl_2021` |
   | `Sale Report.csv` | `product` |
   | `Expense IIGF.csv` | `received_amount`, `expense_detail` |
   | `Cloud Warehouse Compersion Chart.csv` | `warehouse_pricing` |

The app reads the CSVs directly from the zip; there is no need to extract it.

---

## Building the artifacts

```bash
python scripts/build_artifacts.py
```

This takes about 2–3 minutes and:

1. Reproduces the notebook's preprocessing and writes `Data/2_Processed/*.parquet`.
2. Re-runs every statistical test and model experiment with the notebook's exact
   configurations, splits and random seeds (no tuning).
3. Saves tables and metadata to `artifacts/results/` and fitted models to `artifacts/models/` (~240 MB).
4. Prints a **reconciliation report** comparing every reproduced value with the notebook's
   saved output (`tests/reference_values.json`). It is also saved to `artifacts/results/reconciliation/`.

For a quick rebuild of data and statistics only (no model training), run `python scripts/build_artifacts.py --skip-models`.

The descriptive pages also work without this step: if the processed files are missing, the
app runs the preprocessing from the zip on first load (a few seconds). Statistical and model
sections show a message asking you to run the build script.

---

## Running the dashboard

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>. Paths are resolved relative to the code, so you can also
launch it from another folder (`streamlit run "path/to/Analytical Dashboard/app.py"`).

---

## Running the tests

```bash
pytest
```

| Test file | What it checks |
|---|---|
| `test_pipeline.py` | Dataset shapes, KPIs and descriptive figures equal the notebook's outputs |
| `test_statistics.py` | Chi-square, Kruskal-Wallis, Dunn and Spearman results equal the notebook's |
| `test_app.py` | Every page renders headlessly without errors; empty filters warn; filters change KPIs; the experiment toggle works |

Tests that need the data are skipped automatically if the zip is missing.

---

## Reproducibility

The build script reproduces **171 of 207** checked notebook values exactly:

| Area | Result |
|---|---|
| Datasets, KPIs, descriptive figures | All exact |
| Hypothesis tests and feature screening | All exact |
| Experiment 1.1 (6 models) | All metrics exact |
| Experiment 1: unweighted and balanced Logistic Regression, Random Forest | Exact, including confusion matrices |
| Customer segmentation, product price clustering | Exact (silhouette, Davies-Bouldin, cluster sizes and profiles, PCA) |
| Forecasting: Naive, Moving Average, ADF | Exact |
| Experiment 1: XGBoost (2 models) | Small drift (≤ 0.002 on most metrics) |
| Experiment 1: SMOTE pipelines (4 models) | Drift (e.g. SMOTE Logistic Regression recall 0.30 vs 0.41) |
| Forecasting: SES, Holt | MAE within 0.03% and 0.3% |

The remaining differences come from the platform (the notebook ran on Kaggle/Linux),
not from the method. XGBoost's multithreaded floating-point sums differ by OS, and
SMOTE breaks ties between identical one-hot neighbours differently. None of the
differences changes a conclusion: every cancellation model stays at ROC-AUC ≈ 0.51–0.57, and
the naive forecast stays best.

The dashboard shows the **re-built** metrics, so they agree with the confusion matrices and
predictions shown next to them. The notebook's printed metrics appear in a side-by-side
column, and drifting models are flagged.

---

## Methodology notes

These behaviours of the source notebook are **reproduced on purpose**, not "fixed":

1. **Amazon cleaning.** Cell 108 re-assigns `amazon = dfs["amazon"]`, discarding earlier
   snake_case renaming, dtype fixes and an `amount = 0` fill for zero-quantity rows. All
   later analyses use the raw column names and keep 7,795 missing `Amount` values, and so
   does this app.
2. **International main table.** Only rows 0–18,634 are the transaction table; later rows
   are other tables appended in the same CSV.
3. **International de-duplication.** Rows with the same `date, customer, style, sku, size`
   are treated as duplicates (first kept; rows with any missing key component kept). This
   removes 6,282 rows (33.7%). It is an analytical assumption: there is no transaction ID.
4. **Product inventory.** Empty rows and exact duplicates are removed; `#REF!` becomes
   missing; four SKUs that appear twice with different attributes are kept.
5. **Expense file.** The 8,095 subtotal row is removed; the final unlabelled 5,000 row is the
   recorded summary (gross 8,095 − pending 3,095 = 5,000), following the notebook's
   section 6.5.3. An earlier KPI table that summed it to 10,000 is not used.
6. **No cross-dataset joins.** `sku_standardized` is created per dataset but never used as a
   join key. The only merges are the notebook's own May-2022 ↔ P&L-2021 price comparison.
7. **"Product clustering"** clusters the *May-2022 pricing* SKUs (1,293 with complete
   marketplace prices), not the inventory table. The separate price-change analysis
   ("Pricing → no point" in the notebook) is not shown.
8. **Experiment 1.1** undersamples April to **65:35** (the notebook comment says 70:30; the code
   uses a 35% cancellation rate) and was only evaluated on the May validation month.
9. **Notebook text inconsistency.** The notebook's written summary of the Dunn test says
   24/28 category pairs differ and lists Top–Western Dress as non-significant. Its printed
   Dunn table shows 25/28, with Top–Western Dress significant. The dashboard follows the
   printed result.

---

## Limitations

- *Recorded amount* is not net revenue; cancellations and returns are included.
- Order cancellation models have very weak predictive power and are **not production-ready**.
- International results depend on the de-duplication assumption; April–May 2022 records have
  no SKU, and May 2022 is a partial month.
- Forecasting uses 11 monthly observations and one annual cycle; it is exploratory only.
- Stock is a single snapshot without history; it does not indicate demand.
- The financial files have 4–15 rows and support description only.
- Statistical tests on transaction-level International data treat repeated purchases as
  independent; their p-values are optimistic.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Dataset zip not found" | Place the zip at `Data/1_Raw/E-Commerce Sales Dataset.zip` (exact name). |
| "Results not built yet" | Run `python scripts/build_artifacts.py`, then refresh the page. |
| `pip` fails with `CERTIFICATE_VERIFY_FAILED` | An antivirus or corporate proxy is inspecting HTTPS. Pass its root certificate to pip, e.g. `pip install --cert "C:\ProgramData\AVG\Antivirus\wscert.pem" -r requirements-dev.txt` (AVG), or ask IT for the proxy certificate. |
| `Activate.ps1 cannot be loaded` | See the execution-policy command under [Setup](#setup). |
| Warning about Python version | Use Python 3.12; other versions may change model results slightly. |
| Model numbers differ from this README | Check that installed versions match `requirements.txt` (`pip freeze`), rebuild, and read the reconciliation report. |

---

## Git workflow

The repository is set up so that data, processed files, model artifacts and the virtual
environment can never be committed (see `.gitignore`). Empty folders are kept with
`.gitkeep` files.

First push to a new GitHub repository (create it on GitHub first, without a README):

```bash
git add -A
```

```bash
git status
```

Confirm that no `.zip`, `.parquet`, `.joblib` or `.venv` files are listed, then:

```bash
git commit -m "Initial commit: e-commerce analytics dashboard"
```

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
```

```bash
git push -u origin main
```

Anyone cloning the repository follows [Setup](#setup), [Getting the data](#getting-the-data)
and [Building the artifacts](#building-the-artifacts).
