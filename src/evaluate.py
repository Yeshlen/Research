"""
Evaluate rule-based, anomaly (Isolation Forest, LOF, One-Class SVM, autoencoder) and hybrid risk scoring.

Protocol (no test information is used to tune anything):
  fit         train split, days 0-89   -> anomaly models are fitted here (labels unused)
  validation  train split, days 90-119 -> LOF k, OCSVM nu, hybrid weight w and alert thresholds are chosen here
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
from models import (STAGE_A_FEATURES, STAGE_B_FEATURES, autoencoder, band, hybrid_score,  # noqa: E402
                    isolation_forest, lof, one_class_svm, rule_score)

VAL_START = 90
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)
# name -> (short name used in hybrid labels, factory taking (features, seed, hyperparameter), validation grid)
ANOMALY = {
    "Isolation Forest": ("IF", lambda f, seed, _: isolation_forest(f, seed), [None]),
    "LOF": ("LOF", lambda f, seed, k: lof(f, k, seed), [20, 50, 100, 200]),
    "One-Class SVM": ("OCSVM", lambda f, seed, nu: one_class_svm(f, nu, seed), [0.01, 0.05, 0.1, 0.2]),
    "Autoencoder": ("AE", lambda f, seed, _: autoencoder(f, seed), [None]),
}
MODEL_COLOR = {"Rules": INK2, "Isolation Forest": LEGIT, "LOF": "#1baf7a", "One-Class SVM": "#6250d6",
               "Autoencoder": "#eda100"}
HYBRID_COLOR = SCAM
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
    """Score one stage on validation and test; tune everything on validation only."""
    features = STAGE_A_FEATURES if stage == "A" else STAGE_B_FEATURES
    if stage == "B":
        df = df[df.redeemed == 1]
    fit = df[(df.split == "train") & (df.day < VAL_START)]
    val = df[(df.split == "train") & (df.day >= VAL_START)]
    test = df[df.split == "test"]
    yv = val.is_scam.to_numpy()

    val_s = {"Rules": rule_score(val, stage)[0]}
    test_s = {"Rules": rule_score(test, stage)[0]}
    tuning = []
    for name, (_, make, grid) in ANOMALY.items():
        best = None
        for hp in grid:
            scorer = make(features, seed, hp).fit(fit)
            v = scorer.score(val)
            ap_v = average_precision_score(yv, v)
            tuning.append({"stage": stage, "model": name, "param": hp, "val_pr_auc": ap_v})
            if best is None or ap_v > best[0]:
                best = (ap_v, scorer, v)
        val_s[name], test_s[name] = best[2], best[1].score(test)

    hybrids = {}
    for name, (short, _, _) in ANOMALY.items():
        w_auc = {w: average_precision_score(yv, hybrid_score(val_s["Rules"], val_s[name], w)) for w in WEIGHTS}
        w = max(w_auc, key=w_auc.get)
        h = f"Rules + {short}"
        hybrids[h] = {"w": w, "val_pr_auc": w_auc[w]}
        tuning += [{"stage": stage, "model": h, "param": k, "val_pr_auc": v} for k, v in w_auc.items()]
        val_s[h] = hybrid_score(val_s["Rules"], val_s[name], w)
        test_s[h] = hybrid_score(test_s["Rules"], test_s[name], w)

    thresholds = {m: best_f1_threshold(yv, v) for m, v in val_s.items()}
    best_anomaly = max(ANOMALY, key=lambda m: average_precision_score(yv, val_s[m]))
    best_hybrid = max(hybrids, key=lambda h: hybrids[h]["val_pr_auc"])
    return {"test": test, "scores": test_s, "thresholds": thresholds, "hybrids": hybrids, "tuning": tuning,
            "best_anomaly": best_anomaly, "best_hybrid": best_hybrid}


def color(model):
    return MODEL_COLOR.get(model, HYBRID_COLOR)


def scenario_alert_rates(test, alerts):
    """Share of vouchers flagged per scenario: recall for scam types, false-alarm rate for legit types."""
    rows = {}
    for model, pred in alerts.items():
        rows[model] = pd.Series(pred, index=test.index).groupby(test.scenario_type).mean()
    out = pd.DataFrame(rows).reindex(SCENARIO_ORDER)
    out.insert(0, "n_test", test.scenario_type.value_counts().reindex(SCENARIO_ORDER))
    return out


# ---------------------------------------------------------------- figures
def stage_title(stage):
    return "Stage A: at creation" if stage == "A" else "Stage B: at cash-out"


def fig_pr_curves(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, (stage, r) in zip(axes, results.items()):
        y = r["test"].is_scam.to_numpy()
        for model in ["Rules", *ANOMALY, r["best_hybrid"]]:
            s = r["scores"][model]
            p, rec, _ = precision_recall_curve(y, s)
            hyb = model == r["best_hybrid"]
            ax.plot(rec, p, color=color(model), lw=2 if hyb else 1.3,
                    label=f"{model} (AP {average_precision_score(y, s):.2f})")
            if model == "Rules" or hyb:
                m = metrics(y, s, r["thresholds"][model])
                ax.plot(m["recall"], m["precision"], "o", color=color(model), ms=5, mec="white", zorder=5)
        ax.axhline(y.mean(), color=MUTED, lw=0.8, ls="--")
        ax.text(0.01, y.mean() + 0.015, f"base rate {y.mean():.1%}", color=MUTED, ha="left", va="bottom",
                fontsize=7.5)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", title=stage_title(stage))
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("Precision")
    fig.text(0.5, -0.04, "Test split (days 120-179). Hybrid shown: best on validation. "
             "Dots: operating threshold chosen on validation.", ha="center", color=MUTED, fontsize=7.5)
    save(fig, out, "pr_curves")


def fig_scenarios(results, rates, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    y = np.arange(len(SCENARIO_ORDER))
    h = 0.26
    for ax, (stage, r) in zip(axes, results.items()):
        shown = ["Rules", r["best_anomaly"], r["best_hybrid"]]
        for i, model in enumerate(shown):
            ax.barh(y + (i - 1) * h, rates[stage][model], height=h, color=color(model), label=model)
        ax.axhline(4.5, color=AXIS, lw=0.8)
        ax.set(xlim=(0, 1), title=stage_title(stage), xlabel="Share of vouchers flagged")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_yticks(y, [s.replace("_", " ") for s in SCENARIO_ORDER])
    axes[0].invert_yaxis()
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
    for stage, r in results.items():
        test, scores, thresholds = r["test"], r["scores"], r["thresholds"]
        y = test.is_scam.to_numpy()
        print(f"\nStage {stage}: {len(test):,} test vouchers, {y.sum():,} scams. "
              f"Best anomaly model: {r['best_anomaly']}; best hybrid: {r['best_hybrid']} "
              f"(w = {r['hybrids'][r['best_hybrid']]['w']})")
        alerts = {}
        for model, s in scores.items():
            kind = "rules" if model == "Rules" else "hybrid" if model.startswith("Rules +") else "anomaly"
            w = r["hybrids"][model]["w"] if kind == "hybrid" else np.nan
            m = metrics(y, s, thresholds[model])
            rows.append({"stage": stage, "model": model, "kind": kind, "hybrid_w": w,
                         "operating point": "max F1 on validation", **m})
            alerts[model] = s >= thresholds[model]
        m = metrics(y, scores["Rules"], 70)
        rows.append({"stage": stage, "model": "Rules", "kind": "rules", "hybrid_w": np.nan,
                     "operating point": "High band (score >= 70)", **m})
        rates[stage] = scenario_alert_rates(test, alerts)
        if stage == "A":
            bands = fig_bands(test, scores["Rules"], a.out)

    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/tables/model_comparison.csv", index=False)
    pd.DataFrame([t for r in results.values() for t in r["tuning"]]).to_csv(
        f"{a.out}/tables/validation_tuning.csv", index=False)
    label = res.model.str.replace("Rules + ", "Hybrid: Rules + ", regex=False)
    label = label.where(~res["operating point"].str.startswith("High"), label + " ($\\geq$70)")
    label = label.where(res.hybrid_w.isna(), label + " ($w$=" + res.hybrid_w.map("{:.1f}".format) + ")")
    tex = res.assign(Stage=res.stage, Model=label)
    tex = tex.set_index(["Stage"])[["Model", "precision", "recall", "f1", "fpr", "pr_auc", "roc_auc"]]
    tex.columns = ["Model", "Precision", "Recall", "F1", "FPR", "PR-AUC", "ROC-AUC"]
    tex[tex.columns[1:]] = tex[tex.columns[1:]].map(lambda v: f"{v:.3f}")
    write_tex(tex, f"{a.out}/tables/model_comparison.tex",
              "Risk scoring on the test split (days 120--179). Hyperparameters, hybrid weights $w$ and "
              "thresholds chosen on validation days 90--119.", "tab:model-comparison")

    scen = pd.concat(rates, names=["stage"]).round(3)
    scen.to_csv(f"{a.out}/tables/scenario_detection.csv")
    fig_pr_curves(results, a.out)
    fig_scenarios(results, rates, a.out)

    pd.set_option("display.width", 200)
    print("\n" + res[["stage", "model", "hybrid_w", "operating point", "precision", "recall", "f1", "fpr",
                      "pr_auc", "roc_auc"]].round(3).to_string(index=False))
    print("\nValidation tuning (anomaly hyperparameters):")
    tun = pd.DataFrame([t for r in results.values() for t in r["tuning"]])
    print(tun[~tun.model.str.startswith("Rules")].round(3).to_string(index=False))
    print("\nShare flagged per scenario (test):\n" + scen.to_string())
    print("\nRule risk bands, stage A test (count by class):\n" + bands.astype(int).to_string())


if __name__ == "__main__":
    main()
