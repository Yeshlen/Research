"""Apply saved fitting-period models and validation thresholds to individual vouchers."""
import numpy as np

from models import STAGE_A_FEATURES, STAGE_B_FEATURES, hybrid_score, rule_score


def score_vouchers(records, selected, model_name, stage):
    if stage not in ("A", "B"):
        raise ValueError("Unknown scoring stage")
    if model_name not in selected["thresholds"]:
        raise ValueError("Unknown scoring method")
    if stage == "B" and not (records.redeemed == 1).all():
        raise ValueError("Redemption scoring needs a recorded or simulated redemption.")
    fields = STAGE_A_FEATURES if stage == "A" else STAGE_B_FEATURES
    # Truth, scenario and customer/voucher identifiers never reach a scorer.
    inputs = records[fields].copy()
    if not np.isfinite(inputs.astype(float).to_numpy()).all():
        raise ValueError("The selected voucher has incomplete or invalid scoring features.")
    rules, reasons = rule_score(inputs, stage)
    anomaly_name = selected["hybrid_anomaly"] if model_name == "Hybrid" else model_name
    anomaly = None
    if model_name != "Rules":
        anomaly = selected["scorers"][anomaly_name].score(inputs)
    if model_name == "Rules":
        scores, scale = rules, 1.0
    elif model_name == "Hybrid":
        scores, scale = hybrid_score(rules, anomaly, selected["hybrid_weight"]), 100.0
    else:
        scores, scale = anomaly, 100.0
    threshold = float(selected["thresholds"][model_name])
    decisions = []
    for i, (index, row) in enumerate(records.iterrows()):
        decisions.append(dict(voucher_id=int(row.voucher_id), customer_id=int(row.customer_id),
                              amount=float(row.amount), model=model_name, stage=stage,
                              risk=float(scores[i] * scale), threshold=threshold * scale,
                              alert=bool(scores[i] >= threshold), rule_score=float(rules[i]),
                              anomaly_percentile=None if anomaly is None else float(anomaly[i]),
                              reasons=reasons[i], anomaly_model=anomaly_name if anomaly is not None else None))
    return decisions
