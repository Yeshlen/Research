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

## Status

- [x] Dataset design draft (`docs/dataset_design.md`)
- [x] Generator sketch (`src/generator_sketch.py`)
- [ ] Full generator + event log
- [ ] EDA
- [ ] Rule-based, anomaly and hybrid models
- [ ] Dashboard
- [ ] Report
