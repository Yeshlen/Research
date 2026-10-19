"""
Score every test voucher with the deployed model (Rules + One-Class SVM hybrid) for the dashboard, plus the
supervised upper-bound benchmark (needs labels) for comparison.

Uses the same protocol as evaluate.py: the anomaly model is fitted on days 0-89, the hybrid weight,
alert threshold and risk band cut-offs are chosen on validation days 90-119, and days 120-179 are scored.

Band cut-offs on the 0-100 risk score (= 100 x hybrid score), all set on validation:
  Medium   -> lowest score that still keeps validation recall at 80%
  High     -> the alert threshold (max F1 on validation)
  Critical -> lowest score where validation precision reaches 70%
  (raised to 10 points above the alert threshold's own recall / precision when it already exceeds them)

Also times scoring one voucher at a time (stored in the metadata as latency_ms) to back the real-time claim.

Writes data/synthetic/scored_vouchers.csv, data/synthetic/customer_events_test.parquet and
data/synthetic/dashboard_meta.json.

Usage:
    python src/score_all.py --data data/synthetic
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import SUPERVISED, score_stage, split_stage  # noqa: E402
from models import STAGE_A_FEATURES, STAGE_B_FEATURES, hybrid_score, one_class_svm, rule_score  # noqa: E402

MODEL = {"One-Class SVM": ("OCSVM", lambda f, seed, nu: one_class_svm(f, nu, seed), [0.1])}
HYBRID = "Rules + OCSVM"
TOP_ANOMALY = 0.99
CRIT_PRECISION = 0.7


def band_cutoffs(y, s, threshold):
    p, r, t = precision_recall_curve(y, s)
    p, r = p[:-1], r[:-1]
    i = np.searchsorted(t, threshold)
    p_thr, r_thr = p[min(i, len(p) - 1)], r[min(i, len(r) - 1)]
    # targets stay at 80% recall / 70% precision unless the alert threshold already exceeds them
    # (as for the supervised benchmark), so every band keeps a distinct meaning
    rec_target, prec_target = max(0.8, min(r_thr + 0.1, 0.99)), max(CRIT_PRECISION, min(p_thr + 0.1, 0.99))
    medium = t[r >= rec_target].max() if (r >= rec_target).any() else t.min()
    critical = t[p >= prec_target].min() if (p >= prec_target).any() else t.max()
    medium = min(medium, threshold)
    critical = max(critical, threshold)
    return {"Medium": 100 * float(medium), "High": 100 * float(threshold), "Critical": 100 * float(critical)}


def scoring_latency(fit, test, stage, w, n=300, seed=0):
    """Time scoring vouchers one at a time (rules + anomaly model + blend), as a live system would.

    Returns the median and 95th percentile in milliseconds per voucher.
    """
    scorer = one_class_svm(STAGE_A_FEATURES if stage == "A" else STAGE_B_FEATURES, 0.1, seed).fit(fit)
    rows = test.sample(n, random_state=seed)
    ms = []
    for i in range(n):
        row = rows.iloc[[i]]
        t0 = time.perf_counter()
        hybrid_score(rule_score(row, stage)[0], scorer.score(row), w)
        ms.append(1000 * (time.perf_counter() - t0))
    return {"p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95))}


def to_band(score, cut):
    return np.select([score >= cut["Critical"], score >= cut["High"], score >= cut["Medium"]],
                     ["Critical", "High", "Medium"], "Low")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic")
    a = ap.parse_args()
    df = pd.read_csv(f"{a.data}/cardless_features.csv")

    test_all = df[df.split == "test"].copy()
    out = test_all[["voucher_id", "customer_id", "created_at", "day", "amount", "redeemed", "redeem_delay_min",
                    "redeem_dist_usual_km", "recipient_known", "is_self_send", "limit_changed_24h",
                    "scenario_type", "is_scam"]].set_index("voucher_id")
    out["created_at"] = pd.to_datetime(out.created_at)
    out["redeemed_at"] = out.created_at + pd.to_timedelta(out.redeem_delay_min, unit="min")
    meta = {"model": HYBRID, "protocol": "fit days 0-89, tune on 90-119, score 120-179", "stages": {}}

    for stage in ("A", "B"):
        fit, val, test = split_stage(df, stage)
        r = score_stage(fit, val, test, stage, seed=0, anomaly=MODEL, supervised=True)
        risk = 100 * r["scores"][HYBRID]
        cut = band_cutoffs(val.is_scam.to_numpy(), r["val_scores"][HYBRID], r["thresholds"][HYBRID])
        reasons = rule_score(test, stage)[1]
        anomaly = r["scores"]["One-Class SVM"]
        for i, pct in enumerate(anomaly):
            if pct >= TOP_ANOMALY:
                reasons[i] = reasons[i] + [f"unusual overall pattern (anomaly model, top {100 * (1 - pct):.1f}%)"]
        s = stage.lower()
        cols = pd.DataFrame({f"rule_{s}": r["scores"]["Rules"], f"anomaly_{s}": anomaly, f"risk_{s}": risk,
                             f"band_{s}": to_band(risk, cut), f"alert_{s}": risk >= cut["High"],
                             f"reasons_{s}": [" | ".join(x) for x in reasons]}, index=test.voucher_id)
        # supervised upper bound, banded the same way (for comparison in the dashboard only)
        sup = 100 * r["scores"][SUPERVISED]
        sup_cut = band_cutoffs(val.is_scam.to_numpy(), r["val_scores"][SUPERVISED], r["thresholds"][SUPERVISED])
        cols[f"sup_risk_{s}"] = sup
        cols[f"sup_band_{s}"] = to_band(sup, sup_cut)
        cols[f"sup_alert_{s}"] = sup >= sup_cut["High"]
        out = out.join(cols)
        lat = scoring_latency(fit, test, stage, r["hybrids"][HYBRID]["w"])
        meta["stages"][stage] = {"hybrid_w": r["hybrids"][HYBRID]["w"], "cutoffs": cut, "supervised_cutoffs": sup_cut,
                                 "latency_ms": lat}
        print(f"Stage {stage}: one voucher scores in {lat['p50']:.1f} ms (median), {lat['p95']:.1f} ms (95th pct)")
        print(f"Stage {stage}: w={r['hybrids'][HYBRID]['w']}, cut-offs " +
              ", ".join(f"{k} {v:.1f}" for k, v in cut.items()))
        print(pd.crosstab(out[f"band_{s}"], out.is_scam).reindex(["Low", "Medium", "High", "Critical"]))
        print("Supervised:", pd.crosstab(out[f"sup_band_{s}"], out.is_scam).reindex(["Low", "Medium", "High", "Critical"]))

    out.reset_index().to_csv(f"{a.data}/scored_vouchers.csv", index=False)
    with open(f"{a.data}/dashboard_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # events of test customers, for the customer drill-down (keeps the dashboard fast)
    ev = pd.read_csv(f"{a.data}/events_log.csv", low_memory=False)
    ev = ev[ev.customer_id.isin(out.customer_id.unique())]
    ev.to_parquet(f"{a.data}/customer_events_test.parquet", index=False)
    print(f"Wrote {len(out):,} scored vouchers and {len(ev):,} events")


if __name__ == "__main__":
    main()
