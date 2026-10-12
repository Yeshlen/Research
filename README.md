# COS700: Real-Time Behavioural Risk Scoring for Cardless Cash Withdrawal Fraud

University of Pretoria COS700 research project (supervisor: Prof Jan Eloff).

Prototype that simulates mobile banking events, extracts behavioural features, scores cardless withdrawal vouchers (CashSend / eWallet / Instant Money style) and raises alerts for likely social-engineering (vishing) scams.

## Layout

| Path | Contents |
|---|---|
| `src/` | Synthetic data generator, feature extraction, rule-based scoring, anomaly models, evaluation |
| `data/synthetic/` | Generated datasets (not committed; regenerate from the seed) |
| `notebooks/` | EDA and experiment notebooks |
| `tests/` | Sanity and constraint tests for the generator |
| `report/` | LaTeX report (course template), bibliography and figures |
| `docs/` | Dataset design and report planning notes |
| `inputs/` | Submitted proposal and the report guideline slides |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Generate the dataset

```bash
python src/simulate.py      # customers -> event log (data/synthetic/events_log.csv), ~1 min
python src/features.py      # event log -> feature table (data/synthetic/cardless_features.csv), ~1 min
python src/validate.py      # constraint checks, summary, "too easy" check
python src/eda.py           # EDA figures -> report/figures, tables -> report/tables
python src/evaluate.py      # rules, IF, LOF, OCSVM, autoencoder, hybrids on the test split (~2 min)
pytest -q                   # generator and scoring tests on a small population
```

All parameters live in `config/generator.yaml`; the same seed gives the same dataset. Column meanings: `docs/data_dictionary.md`.

## Status

- [x] Dataset design draft (`docs/dataset_design.md`)
- [x] Generator v1: event log, streaming features, validation, tests
- [x] EDA figures and findings (`docs/eda_findings.md`)
- [x] Rule-based, Isolation Forest and hybrid scoring (`docs/model_results.md`)
- [x] LOF, One-Class SVM, autoencoder and their hybrids
- [ ] Learning curve and sensitivity analysis
- [ ] Dashboard
- [ ] Report
