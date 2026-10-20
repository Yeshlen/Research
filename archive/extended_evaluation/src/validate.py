"""
Sanity checks for a generated dataset.

1. Domain constraints (no impossible rows)
2. Summary: size, prevalence, vouchers per customer, scenario counts per split
3. "Too easy" check: a depth-3 decision tree trained on the train split. If it reaches near-perfect
   precision and recall on the test split, the scenarios need more overlap.

Usage:
    python src/validate.py --data data/synthetic
"""
import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

STAGE_A = ["amount", "amount_ratio_to_avg_90d", "amount_ratio_to_max", "amount_to_limit_ratio", "history_count",
           "has_history", "days_since_last_voucher", "vouchers_24h", "amount_24h", "vouchers_7d",
           "vouchers_in_session", "recipient_known", "is_self_send", "recipient_added_in_session",
           "secs_since_login", "screens_before_create", "hour", "hour_deviation", "is_night", "day_of_week",
           "is_payday_window", "limit_changed_24h", "limit_increase_24h", "days_since_device_change",
           "new_device_7d", "password_reset_7d", "account_age_days"]
STAGE_B = STAGE_A + ["redeem_delay_min", "redeem_dist_home_km", "redeem_dist_usual_km", "redeem_hour",
                     "redeem_is_night", "redeem_at_atm"]


def constraint_violations(df, cfg):
    """Return {check name: number of violating rows}. All should be zero."""
    p = cfg["product"]
    red = df["redeemed"] == 1
    return {
        "amount not a multiple of step": int((df["amount"] % p["amount_step"] != 0).sum()),
        "amount above max daily limit": int((df["amount"] > p["max_daily_limit"]).sum()),
        "amount above current limit": int((df["amount_to_limit_ratio"] > 1.0001).sum()),
        "calendar-day total above max limit": int(
            (df.assign(d=pd.to_datetime(df["created_at"]).dt.date).groupby(["customer_id", "d"])["amount"].sum()
             > p["max_daily_limit"]).sum()),
        "redeemed but no delay": int((red & df["redeem_delay_min"].isna()).sum()),
        "not redeemed but has delay": int((~red & df["redeem_delay_min"].notna()).sum()),
        "negative redemption delay": int((df["redeem_delay_min"] < 0).sum()),
        "redeemed after expiry": int((df["redeem_delay_min"] > p["voucher_expiry_days"] * 1440).sum()),
        "recipient known with no history": int(((df["recipient_known"] == 1) & (df["history_count"] == 0)).sum()),
        "self-send flagged as added recipient": int(((df["is_self_send"] == 1) & (df["recipient_added_in_session"] == 1)).sum()),
        "voucher before account opened": int((df["account_age_days"] < 0).sum()),
        "missing label": int(df["is_scam"].isna().sum()),
    }


def too_easy_check(df, features):
    train, test = df[df.split == "train"], df[df.split == "test"]
    X_tr, X_te = train[features].fillna(-1), test[features].fillna(-1)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    tree.fit(X_tr, train.is_scam)
    pred, score = tree.predict(X_te), tree.predict_proba(X_te)[:, 1]
    top = pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False).head(4)
    return {
        "precision": precision_score(test.is_scam, pred, zero_division=0),
        "recall": recall_score(test.is_scam, pred),
        "pr_auc": average_precision_score(test.is_scam, score),
        "top_features": ", ".join(f"{k} ({v:.2f})" for k, v in top.items() if v > 0),
    }


def report(df, cfg):
    lines = []
    per_cust = df.groupby("customer_id").size()
    lines.append(f"vouchers: {len(df):,} from {df.customer_id.nunique():,} customers; "
                 f"scam: {df.is_scam.sum():,} ({df.is_scam.mean():.2%}); redeemed: {df.redeemed.mean():.1%}")
    lines.append("vouchers per customer: " + ", ".join(
        f"{k} {v:.0f}" for k, v in per_cust.describe(percentiles=[.25, .5, .75, .95]).items() if k != "count")
        + f"; share with 1-2: {(per_cust <= 2).mean():.0%}; share with 50+: {(per_cust >= 50).mean():.1%}")
    counts = df.pivot_table(index="scenario_type", columns="split", values="voucher_id", aggfunc="count", fill_value=0)
    lines.append("scenario counts by split:\n" + counts.to_string())
    lines.append("constraint violations (all should be 0):")
    for k, v in constraint_violations(df, cfg).items():
        lines.append(f"  {'OK  ' if v == 0 else 'FAIL'} {k}: {v}")
    for name, feats, sub in [("stage A (creation)", STAGE_A, df),
                             ("stage B (redemption)", STAGE_B, df[df.redeemed == 1])]:
        r = too_easy_check(sub, feats)
        lines.append(f"too-easy check, depth-3 tree, {name}: precision {r['precision']:.2f}, "
                     f"recall {r['recall']:.2f}, PR-AUC {r['pr_auc']:.2f}; splits on {r['top_features']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/generator.yaml")
    ap.add_argument("--data", default="data/synthetic")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    df = pd.read_csv(f"{a.data}/cardless_features.csv")
    print(report(df, cfg))


if __name__ == "__main__":
    main()
