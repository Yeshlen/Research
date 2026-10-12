"""
Robustness experiments for the risk scoring models.

  1. Bootstrap confidence intervals: resample test customers to put a 95% interval on each PR-AUC.
  2. Learning curve: train (fit + validation) on a fraction of customers, always test on the full test split.
  3. Model seed: refit the anomaly models with different random seeds.
  4. Dataset seed: rerun on datasets regenerated with other simulator seeds (data/seeds/s*).
  5. Limit-raise ablation: remove the assumption-driven limit-raise signal from every model.
  6. Prevalence: subsample test scam episodes to rarer base rates (thresholds kept from validation).
  7. Rule weights: leave-one-rule-out, and random +/-50% perturbation of every rule weight.

The same protocol as evaluate.py applies: anything tuned is tuned on validation days 90-119.

Usage:
    python src/experiments.py --data data/synthetic --seeds-dir data/seeds --out report
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from eda import INK2, LEGIT, MUTED, SCAM, plt, save  # noqa: E402
from evaluate import ANOMALY, metrics, run_stage, score_stage, split_stage  # noqa: E402
from models import one_class_svm, rule_hits  # noqa: E402

# Hyperparameters fixed at the values evaluate.py picked on validation, so repeated runs stay fast
FIXED = {"Isolation Forest": ANOMALY["Isolation Forest"],
         "One-Class SVM": ("OCSVM", lambda f, seed, nu: one_class_svm(f, nu, seed), [0.1])}
TRACKED = ["Rules", "Isolation Forest", "One-Class SVM", "Rules + IF", "Rules + OCSVM"]
COLOR = {"Rules": INK2, "Isolation Forest": LEGIT, "One-Class SVM": "#6250d6", "Rules + IF": "#1baf7a",
         "Rules + OCSVM": SCAM}
STAGES = ("A", "B")
LIMIT_COLS = ["limit_changed_24h", "limit_increase_24h"]


def pr_aucs(r, models=TRACKED):
    y = r["test"].is_scam.to_numpy()
    return {m: average_precision_score(y, r["scores"][m]) for m in models}


def stage_name(stage):
    return "A: creation" if stage == "A" else "B: cash-out"


# ---------------------------------------------------------------- 1. bootstrap
def bootstrap(r, stage, reps, rng):
    test = r["test"]
    y = test.is_scam.to_numpy()
    cust = test.customer_id.to_numpy()
    ids = np.unique(cust)
    rows_of = pd.Series(np.arange(len(test))).groupby(cust).apply(np.array)
    out = []
    for m, s in r["scores"].items():
        vals = []
        for _ in range(reps):
            idx = np.concatenate(rows_of.loc[rng.choice(ids, len(ids))].to_numpy())
            vals.append(average_precision_score(y[idx], s[idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out.append({"stage": stage, "model": m, "pr_auc": average_precision_score(y, s), "ci_low": lo, "ci_high": hi})
    return out


# ---------------------------------------------------------------- 2. learning curve
def learning_curve(df, fracs, seeds):
    rows = []
    customers = df.customer_id.unique()
    for stage in STAGES:
        fit, val, test = split_stage(df, stage)
        for frac in fracs:
            for seed in seeds:
                keep = set(np.random.default_rng(seed).choice(customers, int(frac * len(customers)), replace=False))
                f, v = fit[fit.customer_id.isin(keep)], val[val.customer_id.isin(keep)]
                r = score_stage(f, v, test, stage, seed, FIXED)
                for m, ap in pr_aucs(r).items():
                    rows.append({"stage": stage, "frac": frac, "seed": seed, "train_vouchers": len(f) + len(v),
                                 "val_scams": int(v.is_scam.sum()), "model": m, "pr_auc": ap})
                print(f"  learning curve {stage} {frac:.2f} seed {seed}: "
                      f"{len(f) + len(v):,} train vouchers, Rules+OCSVM {rows[-1]['pr_auc']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6. prevalence
def prevalence(r, stage, targets, reps, rng):
    test, y = r["test"], r["test"].is_scam.to_numpy()
    episodes = test.loc[test.is_scam == 1, "episode_id"].unique()
    n_legit = int((y == 0).sum())
    rows = []
    for target in targets:
        for rep in range(reps):
            # keep whole scam episodes until the scam share reaches the target
            order = rng.permutation(episodes)
            sizes = test[test.is_scam == 1].groupby("episode_id").size().reindex(order)
            need = target / (1 - target) * n_legit
            kept = order[: max(1, int(np.searchsorted(sizes.cumsum().to_numpy(), need)) + 1)]
            mask = (y == 0) | test.episode_id.isin(kept).to_numpy()
            for m in ("Rules", "Rules + OCSVM"):
                s = r["scores"][m][mask]
                met = metrics(y[mask], s, r["thresholds"][m])
                rows.append({"stage": stage, "target": target, "rep": rep, "model": m,
                             "prevalence": y[mask].mean(), "precision": met["precision"],
                             "recall": met["recall"], "pr_auc": met["pr_auc"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 7. rule weights
def rule_weights(df, stage, draws, rng):
    _, _, test = split_stage(df, stage)
    y = test.is_scam.to_numpy()
    hits, rules = rule_hits(test, stage)
    pts = np.array([r[1] for r in rules], dtype=float)
    base = average_precision_score(y, np.minimum(hits @ pts, 100))
    ablation = []
    for j, rule in enumerate(rules):
        w = pts.copy()
        w[j] = 0
        ablation.append({"stage": stage, "rule": rule[0], "points": rule[1], "fires_scam": hits[y == 1, j].mean(),
                         "fires_legit": hits[y == 0, j].mean(),
                         "delta_pr_auc": average_precision_score(y, np.minimum(hits @ w, 100)) - base})
    perturbed = [average_precision_score(y, np.minimum(hits @ (pts * rng.uniform(0.5, 1.5, len(pts))), 100))
                 for _ in range(draws)]
    return base, pd.DataFrame(ablation), np.array(perturbed)


# ---------------------------------------------------------------- figures
def fig_learning_curve(lc, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, stage in zip(axes, STAGES):
        d = lc[lc.stage == stage]
        for m in TRACKED:
            g = d[d.model == m].groupby("frac").agg(n=("train_vouchers", "mean"), mean=("pr_auc", "mean"),
                                                    lo=("pr_auc", "min"), hi=("pr_auc", "max"))
            ax.plot(g.n, g["mean"], "-o", color=COLOR[m], lw=1.5, ms=3, label=m)
            ax.fill_between(g.n, g.lo, g.hi, color=COLOR[m], alpha=0.15, lw=0)
        ax.set(xscale="log", xlabel="Training vouchers (fit + validation)", title=f"Stage {stage_name(stage)}",
               ylim=(0, 0.6))
    axes[0].set_ylabel("Test PR-AUC")
    axes[1].legend(loc="lower right", fontsize=7)
    fig.text(0.5, -0.04, "Band: min-max over 3 customer samples. Test set fixed at the full test split.",
             ha="center", color=MUTED, fontsize=7.5)
    save(fig, out, "learning_curve")


def fig_sensitivity(prev, ablation, perturbed, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.25, 0.9]})
    ax = axes[0]
    for (stage, m), g in prev.groupby(["stage", "model"]):
        g = g.groupby("target").precision.mean()
        ax.plot(g.index * 100, g, "-o" if stage == "A" else "--o", ms=3, lw=1.4,
                color=SCAM if m.startswith("Rules +") else INK2, label=f"{m} ({stage})")
    ax.set(xscale="log", xlabel="Scam prevalence (%)", ylabel="Precision at fixed threshold",
           title="Rarer scams", ylim=(0, 1))
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[1]
    a = ablation.pivot(index="rule", columns="stage", values="delta_pr_auc").sort_values("A")
    yy = np.arange(len(a))
    ax.barh(yy - 0.2, a["A"], 0.4, color=LEGIT, label="A: creation")
    ax.barh(yy + 0.2, a["B"].fillna(0), 0.4, color=SCAM, label="B: cash-out")
    ax.set_yticks(yy, [r.replace("_", " ") for r in a.index], fontsize=7)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="Change in rules PR-AUC", title="Leave one rule out")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6.5, loc="lower left")

    ax = axes[2]
    for stage, color in zip(STAGES, (LEGIT, SCAM)):
        base, p = perturbed[stage]
        ax.hist(p, bins=25, color=color, alpha=0.6, label=stage_name(stage))
        ax.axvline(base, color=color, lw=1.2)
    ax.set(xlabel="Rules PR-AUC", title="Weights ×U(0.5, 1.5)", yticks=[])
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, out, "sensitivity")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--seeds-dir", default="data/seeds")
    ap.add_argument("--out", default="report")
    ap.add_argument("--bootstrap", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    df = pd.read_csv(f"{a.data}/cardless_features.csv")
    tables = f"{a.out}/tables"
    pd.set_option("display.width", 200)

    print("Main run (all models, seed 0)")
    main_runs = {s: run_stage(df, s, 0) for s in STAGES}

    # 1. bootstrap CIs
    ci = pd.DataFrame([row for s in STAGES for row in bootstrap(main_runs[s], s, a.bootstrap, rng)])
    ci.to_csv(f"{tables}/bootstrap_ci.csv", index=False)
    print("\n95% bootstrap intervals (customers resampled):\n" + ci.round(3).to_string(index=False))

    # 2. learning curve
    print("\nLearning curve")
    lc = learning_curve(df, [0.02, 0.05, 0.1, 0.25, 0.5, 1.0], [0, 1, 2])
    lc.to_csv(f"{tables}/learning_curve.csv", index=False)
    fig_learning_curve(lc, a.out)
    print(lc.groupby(["stage", "frac", "model"]).pr_auc.mean().unstack().round(3).to_string())

    # 3-5. robustness of the headline numbers
    rows = []

    def record(setting, value, runs):
        for s, r in runs.items():
            for m, v in pr_aucs(r).items():
                rows.append({"setting": setting, "value": value, "stage": s, "model": m, "pr_auc": v})

    record("baseline", "dataset seed 42, model seed 0", main_runs)
    for seed in (1, 2, 3, 4):
        record("model seed", seed, {s: run_stage(df, s, seed, FIXED) for s in STAGES})
    for path in sorted(glob.glob(f"{a.seeds_dir}/s*/cardless_features.csv")):
        d = pd.read_csv(path)
        name = os.path.basename(os.path.dirname(path))[1:]
        print(f"  dataset seed {name}: {len(d):,} vouchers, {d.is_scam.mean():.2%} scams")
        record("dataset seed", name, {s: run_stage(d, s, 0, FIXED) for s in STAGES})
    no_limit = df.assign(**{c: 0 for c in LIMIT_COLS})
    record("no limit-raise signal", "rule and features removed", {s: run_stage(no_limit, s, 0, FIXED) for s in STAGES})
    rob = pd.DataFrame(rows)
    rob.to_csv(f"{tables}/robustness.csv", index=False)
    print("\nRobustness (test PR-AUC):\n" + rob.pivot_table(index=["setting", "value", "stage"], columns="model",
                                                           values="pr_auc").round(3).to_string())

    # 6. prevalence
    prev = pd.concat([prevalence(main_runs[s], s, [0.001, 0.0025, 0.005, 0.01], 20, rng) for s in STAGES])
    full = pd.DataFrame([{"stage": s, "target": main_runs[s]["test"].is_scam.mean(), "rep": 0, "model": m,
                          "prevalence": main_runs[s]["test"].is_scam.mean(),
                          **{k: v for k, v in metrics(main_runs[s]["test"].is_scam.to_numpy(), main_runs[s]["scores"][m],
                                                      main_runs[s]["thresholds"][m]).items()
                             if k in ("precision", "recall", "pr_auc")}}
                         for s in STAGES for m in ("Rules", "Rules + OCSVM")])
    prev = pd.concat([prev, full])
    prev.to_csv(f"{tables}/prevalence.csv", index=False)
    print("\nPrevalence (mean over 20 draws):\n" + prev.groupby(["stage", "model", "target"])
          [["prevalence", "precision", "recall", "pr_auc"]].mean().round(4).to_string())

    # 7. rule weights
    abl, perturbed = [], {}
    for s in STAGES:
        base, ab, p = rule_weights(df, s, 500, rng)
        abl.append(ab)
        perturbed[s] = (base, p)
        print(f"\nRules stage {s}: PR-AUC {base:.3f}; weights x U(0.5,1.5): median {np.median(p):.3f}, "
              f"5-95% [{np.percentile(p, 5):.3f}, {np.percentile(p, 95):.3f}]")
    abl = pd.concat(abl)
    abl.to_csv(f"{tables}/rule_ablation.csv", index=False)
    print(abl.round(3).to_string(index=False))
    fig_sensitivity(prev, abl, perturbed, a.out)


if __name__ == "__main__":
    main()
