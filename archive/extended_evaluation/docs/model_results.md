# Model Results v2: rules, four anomaly models, hybrids (dataset v1.1, seed 42)

Regenerate with `python src/evaluate.py` (~2 min). Figures: `report/figures/pr_curves`, `scenario_detection`, `risk_bands`.
Tables: `report/tables/model_comparison.csv/.tex`, `scenario_detection.csv`, `validation_tuning.csv`.

## Protocol
- **Fit** (train, days 0–89): anomaly models are fitted here without labels. Each score becomes a percentile against these rows, so all models share a 0–1 scale.
- **Validation** (train, days 90–119): LOF `k`, OCSVM `nu`, each hybrid weight `w` (grid 0–1, best PR-AUC) and each alert threshold (max F1) are chosen here.
- **Test** (days 120–179): every number below. Test labels tune nothing.
- Stage A scores all 83,947 test vouchers at creation. Stage B re-scores the 80,880 redeemed ones at cash-out. All 1,162 test scams were redeemed.
- Rule weights are hand-set from the scam modus operandi (`src/models.py`) and not fitted to labels. Rules are also reported at the proposal's High band (score ≥ 70).

## Models
| Model | Setup | Why it is in the comparison |
|---|---|---|
| Rules | 13 weighted red-flag rules, score 0–100, alert reasons, Low/Medium/High/Critical bands | Interpretable baseline banks already use |
| Isolation Forest | 300 trees, 256-sample subtrees, log-scaled features | Isolation-based, scale-free, standard for fraud anomaly detection (Liu et al., 2008) |
| LOF | novelty mode, `k` from {20, 50, 100, 200} → 100; fitted on 20k sampled rows | Density-based local outliers (Breunig et al., 2000) |
| One-Class SVM | RBF kernel, `nu` from {0.01, 0.05, 0.1, 0.2} → 0.1; fitted on 10k sampled rows | Boundary-based novelty detection (Schölkopf et al., 2001) |
| Autoencoder | MLP 32-8-32, 40 epochs, mean squared reconstruction error | Reconstruction-based, common in recent fraud work |
| Hybrid | `w·rules/100 + (1−w)·anomaly percentile`, one per anomaly model | The proposal's combined design |
| Supervised GBM (benchmark) | Histogram gradient boosting, 300 iterations, trained **with labels** on days 0–89 | Upper bound: what the same features achieve once confirmed scams exist. Not part of the proposed system |

LOF, OCSVM and the autoencoder use standardised features. They also drop `amount_to_limit_ratio` and `recipient_added_in_session`, which nearly duplicate `amount` and `recipient_known` (EDA |r| ≈ 0.9) and would otherwise be double-counted in distances.

## Test results (base rate 1.4%)

| Stage | Model | Precision | Recall | F1 | FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| A | Rules | 0.51 | 0.32 | 0.39 | 0.4% | 0.30 | 0.92 |
| A | Isolation Forest | 0.40 | 0.31 | 0.35 | 0.7% | 0.30 | 0.90 |
| A | LOF | 0.29 | 0.28 | 0.29 | 1.0% | 0.20 | 0.77 |
| A | One-Class SVM | 0.38 | 0.38 | 0.38 | 0.9% | 0.28 | 0.89 |
| A | Autoencoder | 0.14 | 0.10 | 0.12 | 0.9% | 0.06 | 0.70 |
| A | Rules + IF (w=0.1) | 0.44 | 0.40 | 0.42 | 0.7% | 0.36 | 0.91 |
| A | Rules + LOF (w=0.7) | 0.38 | 0.47 | 0.42 | 1.1% | 0.36 | 0.92 |
| A | **Rules + OCSVM (w=0.1)** | 0.46 | **0.54** | **0.49** | 0.9% | **0.42** | 0.91 |
| A | Rules + AE (w=0.8) | 0.35 | 0.45 | 0.39 | 1.2% | 0.34 | 0.92 |
| A | Rules ≥ 70 | 0.62 | 0.18 | 0.28 | 0.2% | | |
| A | *Supervised GBM (uses labels)* | *0.89* | *0.63* | *0.74* | *0.1%* | *0.81* | *0.99* |
| B | Rules | 0.64 | 0.38 | 0.48 | 0.3% | 0.42 | 0.94 |
| B | Isolation Forest | 0.32 | 0.36 | 0.34 | 1.1% | 0.28 | 0.90 |
| B | LOF | 0.34 | 0.35 | 0.35 | 1.0% | 0.26 | 0.82 |
| B | One-Class SVM | 0.32 | 0.44 | 0.37 | 1.4% | 0.29 | 0.90 |
| B | Autoencoder | 0.20 | 0.13 | 0.16 | 0.8% | 0.08 | 0.73 |
| B | Rules + IF (w=0.5) | 0.49 | 0.53 | 0.51 | 0.8% | 0.48 | 0.95 |
| B | Rules + LOF (w=0.7) | 0.53 | 0.51 | 0.52 | 0.7% | 0.49 | 0.95 |
| B | **Rules + OCSVM (w=0.3)** | 0.56 | **0.54** | **0.55** | 0.6% | **0.51** | 0.94 |
| B | Rules + AE (w=0.9) | 0.60 | 0.40 | 0.48 | 0.4% | 0.45 | 0.95 |
| B | Rules ≥ 70 | 0.42 | 0.52 | 0.47 | 1.1% | | |
| B | *Supervised GBM (uses labels)* | *0.89* | *0.85* | *0.87* | *0.2%* | *0.93* | *1.00* |

The hybrid Rules + OCSVM was also the best on validation at both stages, so choosing it involved no test data.

## Findings
1. **Every hybrid beats its parts.** Each hybrid beats both the rules and its own anomaly model at both stages. The best, Rules + OCSVM, raises PR-AUC from 0.30 to 0.42 at creation and from 0.42 to 0.51 at cash-out, and catches 54% of scams at under 1% false positives. The rules and the anomaly models catch different scams (`scenario_detection`), which supports the proposal's hybrid design.
2. **Among the anomaly models alone, Isolation Forest and One-Class SVM are clearly ahead** (PR-AUC 0.28–0.30). LOF is next (0.20–0.26), and it was sensitive to `k`: validation PR-AUC ranged from 0.11 to 0.27. Many vouchers share identical discrete feature values, which distorts local density. That is a known LOF weakness on mixed binary and continuous data.
3. **The autoencoder performs poorly (PR-AUC 0.06–0.08).** Its reconstruction error is dominated by rare but harmless combinations, such as a recent password reset, a new account or night-time use. Most scams, by contrast, use the victim's own phone with ordinary-looking values. Several architectures, scalers, longer training and PCA reconstruction were tried; none passed 0.15 on validation. This is a useful negative result: a model can reconstruct a scam voucher well because the scam is a *coached normal session*, not a malformed record. Even so, in a hybrid it still adds recall over rules alone.
4. **Scoring again at cash-out helps.** It lifts the best F1 from 0.49 to 0.55 and lowers the false positive rate. The redemption rules (a fast cash-out far from the customer's usual area) carry most of the gain, so hybrid weights move towards the rules at stage B.
5. **Subtle scams stay mostly undetected** (5–20% recall across models). They copy the customer's normal amounts and recipients. This is the honest limit of behavioural scoring without confirmed-fraud labels.
6. **Legitimate emergencies are the main false alarms** (about 35–70% flagged). They share the scam pattern of a limit raise, a new recipient and an unusual amount. Hybrids catch more scams partly by flagging more of these. A step-up check (call-back or a cooling-off delay) suits these cases better than a block; this is a discussion point for the report.
7. **The rule bands work as a triage queue.** At stage A, 0.2% of legitimate vouchers reach High or Critical, against 18% of scams. Medium holds 49% of scams and 8% of legitimate vouchers, which is where the anomaly component adds recall.
8. **Absolute numbers depend on the generator.** They show relative performance on a controlled benchmark, not expected bank performance. `limit_changed_24h` in particular is an assumption-driven signal (see `eda_findings.md`).

## Why the scores look low, and the supervised upper bound
- **Accuracy is not reported because it is meaningless here.** At a 1.4% scam rate, a model that calls every voucher legitimate scores 98.6% accuracy (the hybrid scores 99.4%). Precision, recall, F1 and PR-AUC on the scam class are the informative metrics. A random score has a PR-AUC equal to the base rate (0.014), so the hybrid's 0.42–0.51 is 30–36 times better than random.
- **The proposed models never see a label.** That is the premise of the project: banks have few confirmed vishing cases, and new scam scripts appear before labels do. To measure the cost of that, a supervised gradient-boosting model was trained on the same features with the labels from days 0–89. It reaches PR-AUC 0.81 at creation and 0.93 at cash-out (F1 0.74 / 0.87).
- **The supervised result is an upper bound, not a competitor.** On synthetic data it partly learns the generator's own scam rules, which real fraudsters would not follow so consistently, and it needs a labelled history the bank may not have. The report should therefore present: (1) rules + anomaly detection as the label-free system that works from day one, and (2) supervised learning as the path once confirmed scams accumulate. `docs/robustness.md` (label budget) shows how many confirmed scams that takes.
- Both approaches tune thresholds on the validation days, which needs some labelled validation data. This is noted as a limitation.

## Next
A learning curve over dataset size, sensitivity to scam prevalence, rule weights and random seed, then the dashboard.
