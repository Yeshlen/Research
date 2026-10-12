# EDA Findings (dataset v1.1, seed 42)

Figures: `report/figures/` (PDF for LaTeX, PNG previews). Regenerate with `python src/eda.py`.
Numbers below are from the full 20k-customer run; see `docs/validation_v1.txt`.

## Dataset overview
- 216,147 vouchers from 19,078 customers over 180 days (1 Jul – 27 Dec 2025); 2,761 scam vouchers (1.28%).
- Vouchers per customer are long-tailed: median 7, 18% of customers have only 1–2, 2.6% have 50+ (max 200) (`activity`).
- Daily volume shows payday (25th–1st), weekend and December peaks (`activity`, right panel).
- Scenario counts: `scenario_counts`. Every scam type has 300+ vouchers overall and 130+ in the test split.

## Separation between classes (`feature_distributions`, `flag_rates`, `feature_summary_by_class`)
- **Amount**: scam median R1,800 vs legitimate R600, but the distributions overlap heavily below R2,000 (subtle and split scams).
- **Amount vs own average**: scam median 1.8× vs 1.0×. Many scams still sit near 1×, which is the overlap the design needs.
- **Login-to-voucher time** is bimodal for scams (rushed vs coached), so it overlaps legitimate habitual and novice users.
- **Cash-out delay** is the strongest single stage-B signal (scam median 12 min vs 143 min), but 12% of legitimate vouchers are also cashed out within 25 minutes.
- **Limit raised in last 24h**: 33% of scams vs 1% of legitimate vouchers. This is the strongest single correlate (Spearman 0.29). It matches bank warnings, but it is an assumption that should be stated.
- **Recipient added in session**: 68% vs 32%. **Sent to own number**: 32% vs 14% (overlap from habitual self-senders).
- **New device** (6% vs 2%) and **night-time** (6% vs 4%) are weak, as intended: vishing victims use their own phone during the day.

## Correlations (`correlation_heatmap`)
- No single feature correlates strongly with `is_scam` (max |r| = 0.29), so detection needs a combination of signals. This supports rule and hybrid scoring.
- Expected redundancy: `amount`–`amount_to_limit_ratio` (0.9) and `recipient_known`–`recipient_added_in_session` (−0.9). Drop or justify one of each pair for distance-based models (LOF, OCSVM).

## Benford's Law (`benford`)
- First-digit MAD = 0.030, which Nigrini (2012) would class as non-conforming. The deficit is at digit 2 and the excess at 4–9.
- This is expected, not a defect. Amounts are bounded by the R2,500/R5,000 limits, cluster around typical amounts of R300–R1,000 and are rounded to R50/R100. Benford assumes data spanning several orders of magnitude.
- Report wording: Benford was used as an exploratory check only (see the design doc); it confirmed the amounts are not uniform and show the expected first-digit decay, while product limits explain the deviation.

## Generator changes made because of EDA (v1 → v1.1)
1. Legitimate limit raises were almost absent (0% within 24h), which made `limit_changed_24h` a near-perfect scam flag. Added in-session limit raises when a legitimate voucher exceeds the limit (`p_raise_on_demand`), and raised the background rate.
2. Scam amounts used uniform bands, which produced artificial blocks at R2,000–2,500 and R4,000–5,000. They now use Beta/log-normal shapes.
3. Scam cash-out delay was uniform 2–25 min, with hard edges. It is now log-normal (median 10 min).

Effect: the depth-3 "too easy" tree dropped from PR-AUC 0.36 to 0.18 at creation, so the dataset is harder and more realistic.
