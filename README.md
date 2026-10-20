# COS700: Behavioural Risk Scoring for Cardless Cash Withdrawal Fraud

University of Pretoria COS700 research project (supervisor: Prof Jan Eloff).

A prototype that simulates mobile banking events, extracts behavioural features and scores cardless withdrawal vouchers for possible social-engineering scams.

## Setup and notebook workflow

Create a Python environment and install `requirements.txt`. Select that environment as the notebook kernel, then run all cells in each notebook in order:

| Notebook | What it does |
|---|---|
| `notebooks/01_data_and_eda.ipynb` | Prepare/reuse data, check records, describe scenarios and draw three EDA plots |
| `notebooks/02_fit_models.ipynb` | Create and fit models; choose settings, alert thresholds and one hybrid using validation records |
| `notebooks/03_results_and_dashboard.ipynb` | Evaluate fixed choices, draw results plots and export dashboard scores |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Notebook 01 reuses an existing dataset. Set its `REGENERATE` flag only when rebuilding the simulator data. Data files and fitted-model binaries are excluded from Git; if either is absent, run the preceding notebooks. Outputs are saved under `report/core/`, with fitted models in `report/core/models/`.

## Revised comparison

Exactly five approaches at voucher creation and cash-out: rules, Isolation Forest, LOF, One-Class SVM, and one equal-weight hybrid combining rules with the best validation anomaly model. Fit 90 days, validate 30 days, test 60 days. Creation uses creation dates; cash-out uses cash-out dates. Model settings and alert thresholds are selected by validation F1 before the test period is scored.

Results report precision, recall, F1, false positive rate, underlying decision counts and scenario rates. The anomaly models fit without labels, but validation selection uses scam labels. The data is synthetic and was already examined during development.

Full procedure: `docs/core_workflow.md`. Configuration: `config/generator.yaml`. Feature definitions and simulator assumptions: `docs/data_dictionary.md` and `docs/dataset_design.md`.

## Dashboard and checks

After running all three notebooks:

```bash
streamlit run app/dashboard.py
pytest -q
```

The dashboard replays saved test-period scores. It includes an alert queue, customer history and the five-model comparison, using the same selected hybrid and alert threshold as the notebooks. Rule indicators describe checklist conditions; anomaly scores and risk bands are not fraud probabilities. The date filter uses the scoring event at each stage. A ground-truth switch supports demonstration and is off by default.

The notebooks are the main analysis workflow. Reusable Python modules remain in `src/` for simulation, feature extraction, scoring and checks; the application remains in `app/`.

## Earlier analyses

Original extended evaluation scripts and documentation are preserved under `archive/extended_evaluation/`. Figures/tables outside `report/core/` and earlier LaTeX report fragments describe the older experiment. They are historical, not outputs of the revised five-approach evaluation. See `docs/eda_findings.md` and `docs/model_results.md` for the new run.
