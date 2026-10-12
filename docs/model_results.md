# Model Results v1: rules, Isolation Forest, hybrid (dataset v1.1, seed 42)

Regenerate with `python src/evaluate.py`. Figures: `report/figures/pr_curves`, `scenario_detection`, `risk_bands`.
Tables: `report/tables/model_comparison.csv/.tex`, `scenario_detection.csv`.

## Protocol
- **Fit** (train, days 0–89): Isolation Forest is fitted here without labels. Its score becomes a percentile against these rows.
- **Validation** (train, days 90–119): hybrid weight `w` (grid 0–1, best PR-AUC) and each model's alert threshold (max F1) are chosen here.
- **Test** (days 120–179): every number below. Test labels tune nothing.
- Stage A scores all 83,947 test vouchers at creation. Stage B re-scores the 80,880 redeemed ones at cash-out. All 1,162 test scams were redeemed.
- Rule weights are set by hand from the scam modus operandi (`src/models.py`) and are not fitted to labels. Rules are also reported at the proposal's High band (score ≥ 70).

## Test results (base rate 1.4%)

| Stage | Model | Precision | Recall | F1 | FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| A | Rules | 0.51 | 0.32 | 0.39 | 0.4% | 0.30 | 0.92 |
| A | Isolation Forest | 0.40 | 0.31 | 0.35 | 0.7% | 0.30 | 0.90 |
| A | **Hybrid (w=0.1)** | 0.44 | **0.40** | **0.42** | 0.7% | **0.36** | 0.91 |
| A | Rules ≥ 70 | 0.62 | 0.18 | 0.28 | 0.2% | | |
| B | Rules | 0.64 | 0.38 | 0.48 | 0.3% | 0.42 | 0.94 |
| B | Isolation Forest | 0.32 | 0.36 | 0.34 | 1.1% | 0.28 | 0.90 |
| B | **Hybrid (w=0.5)** | 0.49 | **0.53** | **0.51** | 0.8% | **0.48** | **0.95** |
| B | Rules ≥ 70 | 0.42 | 0.52 | 0.47 | 1.1% | | |

## Findings
1. **The hybrid beats either component at both stages.** PR-AUC is 0.36 vs 0.30 at creation and 0.48 vs 0.42 at cash-out. The rules and the anomaly model catch different scams: rules find classic and limit-raise scams, while Isolation Forest finds self-send and split scams that trip no single rule (`scenario_detection`). This supports the proposal's hybrid design.
2. **Cash-out adds a lot.** Stage B lifts hybrid recall from 40% to 53% at a similar FPR. The redemption delay and location rules carry most of the gain, so the hybrid weight moves towards the rules (w from 0.1 to 0.5).
3. **Isolation Forest alone is weakest at stage B.** The redemption features are noisy for legitimate users (12% fast cash-outs, travel), so unsupervised "rareness" is less specific there than the targeted rules.
4. **Subtle scams stay mostly undetected** (6–14% recall). This is by design: they mimic the customer's normal amounts and recipients, and it is the honest limit of behavioural scoring without confirmed-fraud labels.
5. **Legitimate emergencies are the main false alarms** (45–58% flagged). They share the scam pattern of a limit raise, a new recipient and an unusual amount. This is the trade-off the report should discuss: a step-up check (call-back, cooling-off delay) suits these better than a block.
6. **The bands work as a triage queue.** At stage A, 0.2% of legitimate vouchers reach High or Critical, against 18% of scams. Medium holds 49% of scams and 8% of legitimate vouchers, which is where the hybrid adds recall.
7. **Absolute numbers depend on the generator.** They show relative performance on a controlled benchmark, not expected bank performance. `limit_changed_24h` in particular is an assumption-driven signal (see `eda_findings.md`).

## Next
LOF and One-Class SVM (drop one of each redundant pair: `amount`/`amount_to_limit_ratio`, `recipient_known`/`recipient_added_in_session`), an autoencoder, a dataset-size learning curve and sensitivity to the scam prevalence and rule weights.
