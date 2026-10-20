# Revised model results

Source: executed notebooks 02 and 03. Models fit on days 0–89, use validation days 90–119, and test on days 120–179. Stage A follows creation time; B follows cash-out time. The dataset was previously examined during development.

Exactly five methods per stage: rules, Isolation Forest, LOF, One-Class SVM and one equal-weight hybrid. Validation F1 selected LOF with 100 neighbours and SVM with nu=0.1 at both stages. SVM had the highest validation F1 among the anomaly families, so the hybrid uses it at both stages. No labels entered anomaly fitting; labelled validation selected settings and thresholds.

| Stage | Model | Precision | Recall | F1 | FPR (%) | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Rules | 0.513 | 0.319 | 0.394 | 0.425 | 371 | 352 | 791 | 82433 |
| A | Isolation Forest | 0.415 | 0.301 | 0.349 | 0.597 | 350 | 494 | 812 | 82291 |
| A | LOF | 0.281 | 0.288 | 0.284 | 1.038 | 335 | 859 | 827 | 81926 |
| A | One-Class SVM | 0.374 | 0.386 | 0.380 | 0.905 | 448 | 749 | 714 | 82036 |
| A | Hybrid | 0.408 | 0.521 | 0.457 | 1.061 | 605 | 878 | 557 | 81907 |
| B | Rules | 0.639 | 0.380 | 0.476 | 0.311 | 441 | 249 | 721 | 79797 |
| B | Isolation Forest | 0.330 | 0.331 | 0.331 | 0.974 | 385 | 780 | 777 | 79266 |
| B | LOF | 0.385 | 0.331 | 0.356 | 0.768 | 385 | 615 | 777 | 79431 |
| B | One-Class SVM | 0.283 | 0.495 | 0.360 | 1.821 | 575 | 1458 | 587 | 78588 |
| B | Hybrid | 0.550 | 0.540 | 0.545 | 0.641 | 628 | 513 | 534 | 79533 |

The hybrid caught more scams than rules, with more false alerts. Legitimate emergency false-alarm rates were 65.7% at creation and 53.4% at cash-out. Subtle-scam recall was 13.6% and 12.3%. See `report/core/tables/scenario_detection.csv` for all methods and scenario counts.

Stage A test vouchers: 83,947. Stage B: 81,208. Both contained 1,162 scam vouchers. Stage populations and separately selected thresholds differ, so these are not a controlled measurement of adding redemption features alone. No uncertainty or repeated-seed experiments are included in the revised evaluation.

Full settings and software versions: `report/core/run_metadata.json`. Fitted binary models are local and ignored by Git; regenerate them with notebook 02. The dashboard uses notebook 03's scores and fixed thresholds.
