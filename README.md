# COS700: Real-Time Behavioural Risk Scoring for Cardless Cash Withdrawal Fraud

University of Pretoria COS700 research project (supervisor: Prof Jan Eloff).

Prototype that simulates mobile banking events, extracts behavioural features, scores cardless withdrawal vouchers (CashSend / eWallet / Instant Money style) and raises alerts for likely social-engineering (vishing) scams.

## Layout

| Path | Contents |
|---|---|
| `src/` | Synthetic data generator, feature extraction, rule-based scoring, anomaly models, evaluation |
| `data/synthetic/` | Generated datasets (not committed; regenerate from the seed) |
| `app/` | Streamlit monitoring dashboard |
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
python src/experiments.py   # bootstrap CIs, learning curve, seeds, ablations, label budget (~25 min; see docs/robustness.md)
python src/score_all.py     # score the test period with Rules + OCSVM for the dashboard (~1 min)
streamlit run app/dashboard.py
pytest -q                   # generator and scoring tests on a small population
```

All parameters live in `config/generator.yaml`; the same seed gives the same dataset. Column meanings: `docs/data_dictionary.md`.

## Status

- [x] Dataset design draft (`docs/dataset_design.md`)
- [x] Generator v1: event log, streaming features, validation, tests
- [x] EDA figures and findings (`docs/eda_findings.md`)
- [x] Rule-based, Isolation Forest and hybrid scoring (`docs/model_results.md`)
- [x] LOF, One-Class SVM, autoencoder and their hybrids
- [x] Supervised upper-bound benchmark and label-budget curve
- [x] Learning curve and sensitivity analysis (`docs/robustness.md`)
- [x] Dashboard (`app/dashboard.py`): live alert replay, alert queue with risk bands and reasons, customer drill-down, model performance
- [ ] Report

## Dashboard

`streamlit run app/dashboard.py` replays the test period (days 120–179) as a bank analyst would see it:

- **Live alerts:** plays the chosen day back in accelerated time (5 minutes to 1 hour per second). Each voucher is scored when it is created and again when someone tries to cash it out. An alert card appears the moment a score crosses the threshold, showing the band, the reasons and the bank's action (freeze the PIN, cooling-off warning, block the cash-out, or step-up approval in the app). Play, pause, restart and "next alert" controls drive the demo. Scoring one voucher takes about 12–15 ms (measured by `score_all.py`), so the system can score in real time.
- **Alert queue:** choose a day and a scoring point (creation or cash-out) to see that day's alerts. Each shows its risk score, its Low/Medium/High/Critical band and the reasons it was flagged. An evaluation toggle reveals which alerts were real scams.
- **Customer drill-down:** a customer's vouchers coloured by band, plus their recent account events (limit changes, new devices, new recipients, cash-outs).
- **Model switch:** the label-free Rules + OCSVM system, or the supervised benchmark for comparison.
- **Model performance:** the comparison table, band precision, detection per scenario, and the PR, learning-curve and sensitivity figures.

Band cut-offs are set on validation days, never on the days shown: Medium keeps 80% recall, High is the max-F1 alert threshold, and Critical is where precision reaches 70%. Screenshots are in `report/figures/dashboard_*.png`.
