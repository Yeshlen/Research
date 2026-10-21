# Revised evaluation: notebook workflow

Run the three notebooks in numerical order using the repository environment. Their cells contain the EDA, model fitting, validation decisions, test scoring and plotting. The Python modules in `src/` provide reusable simulation, feature extraction, rules and model wrappers, basic checks and shared calculations.

1. **01_data_and_eda.ipynb** reuses the existing seed-42 dataset (or regenerates it when requested), runs the consistency checks, and saves three plots. Class comparisons use activity before day 120. Counts describing the entire dataset are labelled separately.
2. **02_fit_models.ipynb** fits Isolation Forest, LOF and One-Class SVM without scam labels. Rules remain fixed. It uses validation labels to select settings and alert thresholds by F1, then selects one anomaly model for a fixed equal-weight hybrid. It saves the fitted models and choices before test scoring.
3. **03_results_and_dashboard.ipynb** loads those choices, evaluates exactly five approaches per stage on test records, saves comparison/scenario/count tables and figures, and exports the same hybrid scores for the dashboard.

## Periods and choices

- Fitting: days 0–89; validation: 90–119; test: 120–179.
- Stage A uses creation time; Stage B includes redeemed vouchers and uses redemption time. Both use half-open date ranges; events on a boundary enter the later period.
- Features use preceding customer events. Model parameters and standardisation are fitted only in the fitting period. LOF and SVM retain their existing sample caps (20,000 and 10,000).
- Isolation Forest: 300 trees, 256 observations per tree. LOF neighbours: 20, 50, 100, 200. SVM nu: 0.01, 0.05, 0.1, 0.2; RBF kernel.
- Each candidate uses its highest-validation-F1 threshold. Threshold ties choose the higher threshold; equal model-setting results keep the first candidate listed. Equal anomaly-family results prefer IF, LOF, then SVM.
- Hybrid = 0.5 × (rules / 100) + 0.5 × anomaly percentile. The chosen anomaly family may differ by stage. Its threshold is also selected by validation F1.
- Fitting includes both legitimate and scam records but does not use labels. The complete selection procedure does use validation labels.
- Metrics: precision, recall, F1, false positive rate and TP/FP/FN/TN counts. Scenario rates distinguish scam recall from false alarms on legitimate activity.

The dashboard displays saved test scores, uses the scoring date for filtering, and shows the selected hybrid name. High/Critical bands correspond exactly to alerts. Medium and Critical retain the previous validation-based UI targets (80% recall and 70% precision), with ordered cutoffs; these are display categories, not predicted probabilities or guaranteed test performance.

## Outputs and reproducibility

Tables, PNG/PDF plots and run metadata: `report/core/`. Locally saved fitted models: `report/core/models/` (ignored by Git). Dashboard records: `data/synthetic/core_scored_vouchers.csv` and `core_dashboard_meta.json` (also ignored). Run notebook 02 before 03 if the fitted-model file is absent. A fingerprint prevents use of models/results after data, configuration or fitting code changes.

The saved evaluation used simulator seed 42 and model/subsample seed 0. The current generator configuration uses seed 75; it does not alter existing CSVs or results until regeneration is explicitly run. The generated data is not committed. Use the same configuration and software environment when reproducing the run. The existing dataset and previous results were already examined during development; this is a revised evaluation, not a new unseen dataset. Stage populations differ, so changes between stages cannot be attributed solely to adding redemption features.

## Earlier work

`archive/extended_evaluation/` preserves the former Python evaluation entry points and their notes. Earlier plots and tables outside `report/core/` are historical and must not be combined with the revised results. Autoencoder, supervised benchmark, bootstrap, learning curves and sensitivity/ablation experiments are excluded from this workflow.
