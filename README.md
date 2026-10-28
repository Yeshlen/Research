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

## Dataset simulator

The active entry point is `src/simulate.py`. Its generation stages are in `src/simulator/`: customer profiles, ordinary activity, special scenarios, event recording and CSV output. Start with `src/simulator/run.py` and `docs/simulator_guide.md` to follow the process. The original single-file implementation is preserved unchanged in `archived/simulate.py`.

For a small example in a separate folder:

```bash
python src/simulate.py --customers 10 --out data/simulator_demo
python src/features.py --data data/simulator_demo
```

Notebook 01 uses the same `simulator.generate_dataset` function when regeneration is enabled. The current configuration uses seed 75; existing saved evaluation results describe the earlier seed-42 dataset. Regenerating data requires rerunning model fitting and evaluation before comparing new results.

## Revised comparison

Exactly five approaches at voucher creation and redemption: rules, Isolation Forest, LOF, One-Class SVM, and one equal-weight hybrid combining rules with the best validation anomaly model. Fit 90 days, validate 30 days, test 60 days. Creation uses creation dates; redemption uses redemption dates. Model settings and alert thresholds are selected by validation F1 before the test period is scored.

Results report precision, recall, F1, false positive rate, underlying decision counts and scenario rates. The anomaly models fit without labels, but validation selection uses scam labels. The data is synthetic and was already examined during development.

Full procedure: `docs/core_workflow.md`. Configuration: `config/generator.yaml`. Feature definitions and simulator assumptions: `docs/data_dictionary.md` and `docs/dataset_design.md`.

## Dashboard and checks

After running all three notebooks:

```bash
streamlit run app/dashboard.py
pytest -q
```

The dashboard has pages for Rules, Isolation Forest, LOF, One-Class SVM and Hybrid, plus model comparison and independent record generation. Each model can score an existing test voucher or a saved generated example. Run Simulation creates legitimate/scam activity (or a 50/50 random choice), scores it with fixed models and opens a review-alert modal when the saved threshold is crossed. Generated events and voucher features are stored separately under `data/demo/`; they never change the research history or saved test metrics. Creation and redemption remain separate scoring points. Rule indicators provide context for anomaly models, not internal model explanations. See `docs/dashboard_guide.md` for the workflow and limitations.

The notebooks are the main analysis workflow. Reusable Python modules remain in `src/` for simulation, feature extraction, scoring and checks; the application remains in `app/`.

## Earlier analyses

Original extended evaluation scripts and documentation are preserved under `archive/extended_evaluation/`. Figures/tables outside `report/core/` and earlier LaTeX report fragments describe the older experiment. They are historical, not outputs of the revised five-approach evaluation. See `docs/eda_findings.md` and `docs/model_results.md` for the new run.
