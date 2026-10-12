"""
Evaluate rule-based, Isolation Forest and hybrid risk scoring on the synthetic dataset.

Protocol (no test information is used to tune anything):
  fit         train split, days 0-89   -> Isolation Forest is fitted here (labels unused)
  validation  train split, days 90-119 -> hybrid weight w and alert thresholds are chosen here
  test        days 120-179             -> all reported numbers

Stage A scores every voucher at creation. Stage B re-scores redeemed vouchers at cash-out,
adding the redemption features.

Usage:
    python src/evaluate.py --data data/synthetic --out report
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from eda import AXIS, INK2, LEGIT, MUTED, SCAM, plt, save  # noqa: E402  (shared figure style)
from models import (STAGE_A_FEATURES, STAGE_B_FEATURES, band, hybrid_score,  # noqa: E402
                    isolation_forest, rule_score)

VAL_START = 90
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
MODEL_COLOR = {"Rules": INK2, "Isolation Forest": LEGIT, "Hybrid": SCAM}
SCENARIO_ORDER = ["scam_classic", "scam_self_send", "scam_limit_raise", "scam_split", "scam_subtle",
                  "normal", "legit_emergency", "legit_new_phone", "legit_travel"]


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return t[np.argmax(f1)]


def metrics(y, s, thr):
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": thr, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn), "alert_rate": pred.mean(),
            "pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_stage(df, stage, seed):
    """Score one stage. Returns per-model test scores, chosen thresholds and the hybrid weight."""
    features = STAGE_A_FEATURES if stage == "A" else STAGE_B_FEATURES
    if stage == "B":
        df = df[df.redeemed == 1]
    fit = df[(df.split == "train") & (df.day < VAL_START)]
    val = df[(df.split == "train") & (df.day >= VAL_START)]
    test = df[df.split == "test"]

    iforest = isolation_forest(features, seed).fit(fit)
    scores = {}
    for name, part in (("val", val), ("test", test)):
        rule = rule_score(part, stage)[0]
        scores[name] = {"Rules": rule, "Isolation Forest": iforest.score(part)}

    # hybrid weight: best validation PR-AUC
    yv = val.is_scam.to_numpy()
    w_auc = {w: average_precision_score(yv, hybrid_score(scores["val"]["Rules"],
                                                         scores["val"]["Isolation Forest"], w))
             for w in WEIGHTS}
    w = max(w_auc, key=w_auc.get)
    for name in scores:
        scores[name]["Hybrid"] = hybrid_score(scores[name]["Rules"], scores[name]["Isolation Forest"], w)

    thresholds = {m: best_f1_threshold(yv, s) for m, s in scores["val"].items()}
    return test, scores["test"], thresholds, w, w_auc


def scenario_alert_rates(test, alerts):
    """Share of vouchers flagged per scenario: recall for scam types, false-alarm rate for legit types."""
    rows = {}
    for model, pred in alerts.items():
        rows[model] = pd.Series(pred, index=test.index).groupby(test.scenario_type).mean()
    out = pd.DataFrame(rows).reindex(SCENARIO_ORDER)
    out.insert(0, "n_test", test.scenario_type.value_counts().reindex(SCENARIO_ORDER))
    return out


# ---------------------------------------------------------------- figures
def fig_pr_curves(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, (stage, (test, scores, thresholds, _, _)) in zip(axes, results.items()):
        y = test.is_scam.to_numpy()
        for model, s in scores.items():
            p, r, _ = precision_recall_curve(y, s)
            ax.plot(r, p, color=MODEL_COLOR[model], lw=1.6,
                    label=f"{model} (AP {average_precision_score(y, s):.2f})")
            m = metrics(y, s, thresholds[model])
            ax.plot(m["recall"], m["precision"], "o", color=MODEL_COLOR[model], ms=5, mec="white")
        ax.axhline(y.mean(), color=MUTED, lw=0.8, ls="--")
        ax.text(0.01, y.mean() + 0.015, f"base rate {y.mean():.1%}", color=MUTED, ha="left", va="bottom",
                fontsize=7.5)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall",
               title="Stage A: at creation" if stage == "A" else "Stage B: at cash-out")
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylabel("Precision")
    fig.text(0.5, -0.04, "Test split (days 120-179). Dots: operating threshold chosen on validation.",
             ha="center", color=MUTED, fontsize=7.5)
    save(fig, out, "pr_curves")


def fig_scenarios(rates, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    y = np.arange(len(SCENARIO_ORDER))
    h = 0.26
    for ax, (stage, r) in zip(axes, rates.items()):
        for i, model in enumerate(MODEL_COLOR):
            ax.barh(y + (i - 1) * h, r[model], height=h, color=MODEL_COLOR[model], label=model)
        ax.axhline(4.5, color=AXIS, lw=0.8)
        ax.set(xlim=(0, 1), title="Stage A: at creation" if stage == "A" else "Stage B: at cash-out",
               xlabel="Share of vouchers flagged")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [s.replace("_", " ") for s in SCENARIO_ORDER])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.text(0.5, -0.03, "Above the line: scam scenarios (higher is better). "
             "Below: legitimate scenarios (false alarms, lower is better).", ha="center", color=MUTED,
             fontsize=7.5)
    save(fig, out, "scenario_detection")


def fig_bands(test_a, rule_a, out):
    bands = pd.Series([band(s) for s in rule_a], index=test_a.index)
    order = ["Low", "Medium", "High", "Critical"]
    tab = pd.crosstab(bands, test_a.is_scam).reindex(order).fillna(0)
    share = tab / tab.sum()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(order))
    ax.bar(x - 0.2, share[0], 0.4, color=LEGIT, label="Legitimate")
    ax.bar(x + 0.2, share[1], 0.4, color=SCAM, label="Scam")
    ax.set_xticks(x, order)
    ax.set(ylabel="Share of class", title="Rule score risk bands (stage A, test)")
    ax.legend()
    save(fig, out, "risk_bands")
    return tab


# ---------------------------------------------------------------- tables
def write_tex(table, path, caption, label):
    cols = "l" + "r" * (table.shape[1])
    lines = [r"\begin{table}[ht]", r"\centering\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\hline",
             " & ".join([table.index.name or ""] + list(table.columns)) + r" \\", r"\hline"]
    for idx, row in table.iterrows():
        lines.append(" & ".join([str(idx)] + [str(v) for v in row]) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="report")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for d in ("figures", "tables"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    df = pd.read_csv(f"{a.data}/cardless_features.csv")

    results = {s: run_stage(df, s, a.seed) for s in ("A", "B")}
    rows, rates = [], {}
    for stage, (test, scores, thresholds, w, w_auc) in results.items():
        y = test.is_scam.to_numpy()
        print(f"\nStage {stage}: {len(test):,} test vouchers, {y.sum():,} scams. Hybrid w = {w} "
              f"(validation PR-AUC by w: " + ", ".join(f"{k}:{v:.3f}" for k, v in w_auc.items()) + ")")
        alerts = {}
        for model, s in scores.items():
            m = metrics(y, s, thresholds[model])
            rows.append({"stage": stage, "model": model, "operating point": "max F1 on validation", **m})
            alerts[model] = s >= thresholds[model]
        m = metrics(y, scores["Rules"], 70)
        rows.append({"stage": stage, "model": "Rules", "operating point": "High band (score >= 70)", **m})
        rates[stage] = scenario_alert_rates(test, alerts)
        if stage == "A":
            bands = fig_bands(test, scores["Rules"], a.out)

    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/tables/model_comparison.csv", index=False)
    tex = res.assign(Stage=res.stage, Model=res.model + np.where(res["operating point"].str.startswith("High"),
                                                                   " ($\\geq$70)", ""))
    tex = tex.set_index(["Stage"])[["Model", "precision", "recall", "f1", "fpr", "pr_auc", "roc_auc"]]
    tex.columns = ["Model", "Precision", "Recall", "F1", "FPR", "PR-AUC", "ROC-AUC"]
    tex[tex.columns[1:]] = tex[tex.columns[1:]].map(lambda v: f"{v:.3f}")
    write_tex(tex, f"{a.out}/tables/model_comparison.tex",
              "Risk scoring on the test split (days 120--179). Thresholds and hybrid weight chosen on "
              "validation days 90--119.", "tab:model-comparison")

    scen = pd.concat(rates, names=["stage"]).round(3)
    scen.to_csv(f"{a.out}/tables/scenario_detection.csv")
    fig_pr_curves(results, a.out)
    fig_scenarios(rates, a.out)

    pd.set_option("display.width", 160)
    print("\n" + res[["stage", "model", "operating point", "threshold", "precision", "recall", "f1", "fpr",
                      "alert_rate", "pr_auc", "roc_auc"]].round(3).to_string(index=False))
    print("\nShare flagged per scenario (test):\n" + scen.to_string())
    print("\nRule risk bands, stage A test (count by class):\n" + bands.astype(int).to_string())


if __name__ == "__main__":
    main()
