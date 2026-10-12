# Robustness and Sensitivity (dataset v1.1)

Regenerate with `python src/experiments.py` (~10 min; it needs the extra datasets below first).
Figures: `report/figures/learning_curve`, `sensitivity`. Tables: `report/tables/bootstrap_ci.csv`, `learning_curve.csv`, `robustness.csv`, `prevalence.csv`, `rule_ablation.csv`.

Extra datasets for the dataset-seed check (not committed):
```bash
for s in 7 123; do python src/simulate.py --seed $s --out data/seeds/s$s && python src/features.py --data data/seeds/s$s; done
```
Both pass all constraint checks. Their "too easy" tree PR-AUC is 0.12–0.17 (A) and 0.19–0.21 (B), against 0.18 and 0.23 for seed 42.

## 1. Are the differences real? Bootstrap 95% intervals
Test customers are resampled 200 times; whole customers are resampled because their vouchers are correlated.

| Stage | Rules | Isolation Forest | One-Class SVM | Rules + IF | **Rules + OCSVM** |
|---|---|---|---|---|---|
| A | 0.30 [0.27, 0.33] | 0.30 [0.26, 0.33] | 0.28 [0.25, 0.32] | 0.36 [0.32, 0.39] | **0.42 [0.39, 0.46]** |
| B | 0.42 [0.39, 0.45] | 0.28 [0.25, 0.32] | 0.29 [0.26, 0.32] | 0.48 [0.45, 0.52] | **0.51 [0.47, 0.55]** |

At stage A, Rules + OCSVM's interval does not overlap with any single model's. At stage B, Rules + OCSVM and Rules + IF overlap with each other, but both sit clearly above every single model. The autoencoder (0.06 [0.05, 0.08]) is clearly worst.

## 2. How much data is needed? Learning curve
The models were trained on 2–100% of customers (about 2.6k to 130k training vouchers, 3 samples each). The test set was always the full test split.

- **Performance stops changing after about 6k training vouchers.** Rules + OCSVM sits at 0.41–0.43 (A) and 0.50–0.52 (B) from 5% of customers upwards. Only One-Class SVM at 2% drops noticeably (0.22).
- This is expected for unsupervised models. They learn what *normal* looks like, and a few thousand vouchers describe that. Isolation Forest fits each tree on only 256 rows, and One-Class SVM is capped at 10k rows.
- **So the 216k-voucher size is justified by evaluation, not by training.** The test split needs enough scams (1,162 in total, and 150+ per scam type) to give intervals of about ±0.03 PR-AUC. It also needs enough to report detection per scam type and per legitimate edge case. This is the argument for the report: a dataset sized for statistically reliable evaluation of a rare class (1.3%).

## 3. Model and dataset seeds
The anomaly models were refitted with seeds 1–4, and the full pipeline was run on datasets regenerated with simulator seeds 7 and 123.

| Test PR-AUC, Rules + OCSVM | Stage A | Stage B |
|---|---|---|
| Baseline (dataset 42, model 0) | 0.421 | 0.509 |
| Model seeds 1–4 | 0.415–0.423 | 0.510–0.515 |
| Dataset seeds 7, 123 | 0.434, 0.429 | 0.513, 0.519 |

The ranking of models is the same in every run, and the hybrid moves by at most ±0.015. The results are not an artefact of one random draw.

## 4. Without the limit-raise signal
`limit_changed_24h` is the strongest single correlate, and it rests on an assumption (EDA). To test that dependence, the limit-raise rule and both limit features were removed from every model.

| Stage | Rules | Rules + OCSVM |
|---|---|---|
| A | 0.30 → 0.24 | 0.42 → 0.36 |
| B | 0.42 → 0.36 | 0.51 → 0.46 |

Performance falls by about 0.05–0.06 but the conclusions do not change: the hybrid still beats every single model. The finding does not hinge on the assumption.

## 5. Rarer scams (prevalence)
Real cardless scam prevalence is unknown and probably below the simulated 1.4%. Test scam episodes were subsampled to 0.1–1% while keeping the thresholds chosen on validation (20 draws each).

| Prevalence | Rules precision (A / B) | Rules + OCSVM precision (A / B) | Rules + OCSVM recall |
|---|---|---|---|
| 1.4% | 0.51 / 0.64 | 0.46 / 0.56 | 0.54 |
| 1.0% | 0.43 / 0.55 | 0.38 / 0.47 | 0.53–0.54 |
| 0.5% | 0.28 / 0.38 | 0.23 / 0.31 | 0.54–0.55 |
| 0.1% | 0.07 / 0.11 | 0.06 / 0.08 | 0.53–0.54 |

Recall and ROC-AUC do not depend on prevalence, but precision falls roughly in proportion to it. At 0.1%, about 1 alert in 12 is a scam. This is the base-rate effect every fraud system faces. It supports treating a High score as a **step-up** (call-back, cooling-off delay, in-app warning) rather than a block, and it is why the report should quote PR-AUC and precision alongside ROC-AUC. The hybrid's extra recall comes at slightly lower precision than the rules at the same prevalence.

## 6. Rule weights
- **Random perturbation.** Each weight was multiplied by U(0.5, 1.5) in 500 draws. Rules PR-AUC stays at 0.23–0.37 (A, 5–95%; baseline 0.30) and 0.35–0.48 (B; baseline 0.42). The rule baseline is moderately sensitive to its hand-set weights, which is one more reason to combine it with a data-driven component.
- **Leave one rule out** (change in rules PR-AUC):
  - The rules that help most are `fast_cashout` (−0.11 at B), `amount_unusual` (−0.09 to −0.12), `near_limit` (−0.05 to −0.07) and `limit_raised` (−0.06).
  - `new_recipient` (+0.07) and `recipient_added_now` (+0.03) *hurt*. They fire on 68% of scams but also on 24–27% of legitimate vouchers, because people often send to new people. At 20 and 10 points they push too many legitimate vouchers up the ranking. `far_cashout` adds nothing (fires on 44% of legitimate cash-outs).
  - **Report wording:** the weights were set a priori from published scam descriptions and were deliberately *not* changed after seeing these test-set results, since that would leak test information. Learning the weights on the validation slice (for example, a logistic regression on rule hits) is listed as future work.

## Summary for the report
1. The hybrid's advantage holds under bootstrap intervals, refitting with other model seeds, regenerated datasets and removal of the main assumed signal.
2. The unsupervised models need only a few thousand vouchers to train. Dataset size matters for reliable evaluation of a 1% class.
3. Precision depends on the real scam prevalence. At realistic rates, alerts should trigger friction, not blocks.
4. The hand-set rule weights are a weak point: two "new recipient" rules are too noisy at their current weights.
