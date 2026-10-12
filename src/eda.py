"""
Exploratory data analysis figures for the report.

Reads data/synthetic/cardless_features.csv (+ customers_latent.csv for the activity figure)
and writes vector PDFs (for LaTeX) and PNG previews to report/figures/, plus summary tables
to report/tables/.

Usage:
    python src/eda.py --data data/synthetic --out report
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Palette: dataviz reference palette, light mode, two categorical slots + chrome
LEGIT, SCAM = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#184f95", "#6da7ec", "#f0efec", "#ec8b8a", "#b02a2a"])
CLASS_LABEL = {0: "Legitimate", 1: "Scam"}
CLASS_COLOR = {0: LEGIT, 1: SCAM}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
    "legend.labelcolor": INK2,
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, "figures", f"{name}.pdf"))
    fig.savefig(os.path.join(out, "figures", f"{name}.png"))
    plt.close(fig)


def class_legend(ax, **kw):
    handles = [plt.Line2D([], [], color=CLASS_COLOR[k], lw=2, label=CLASS_LABEL[k]) for k in (0, 1)]
    ax.legend(handles=handles, **kw)


# ---------------------------------------------------------------- figures
def fig_scenarios(df, out):
    counts = df.scenario_type.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    colors = [SCAM if s.startswith("scam") else LEGIT for s in counts.index]
    ax.barh(counts.index, counts.values, color=colors, height=0.6, edgecolor="white", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Vouchers (log scale)")
    ax.set_title("Vouchers by scenario type")
    for y, v in enumerate(counts.values):
        ax.text(v * 1.08, y, f"{v:,}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(right=counts.max() * 4)
    ax.grid(axis="y", visible=False)
    class_legend(ax, loc="lower right")
    save(fig, out, "scenario_counts")


def fig_activity(df, latent, out):
    per = df.groupby("customer_id").size()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    bins = np.unique(np.logspace(0, np.log10(per.max() + 1), 30).astype(int))
    ax.hist(per, bins=bins, color=LEGIT, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Vouchers per customer (log scale)")
    ax.set_ylabel("Customers")
    ax.set_title("Activity per customer")
    ax.text(0.97, 0.95, f"median {per.median():.0f}\n{(per <= 2).mean():.0%} have 1-2\n{(per >= 50).mean():.1%} have 50+",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax = axes[1]
    daily = df.assign(date=pd.to_datetime(df.created_at).dt.normalize()).groupby("date").size()
    ax.plot(daily.index, daily.values, color=LEGIT, lw=1.2)
    ax.set_title("Vouchers created per day")
    ax.set_ylabel("Vouchers")
    ax.tick_params(axis="x", rotation=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.text(0.02, 0.95, "spikes: payday (25th-1st), weekends,\nDecember", transform=ax.transAxes,
            va="top", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, out, "activity")


def density_panel(ax, df, col, title, xlabel, log=False, clip=None, bins=50):
    data = {k: df.loc[df.is_scam == k, col].dropna() for k in (0, 1)}
    allv = pd.concat(data.values())
    if clip is not None:
        allv = allv.clip(*clip)
    if log:
        lo = max(allv[allv > 0].min(), 1e-2)
        edges = np.logspace(np.log10(lo), np.log10(allv.max()), bins)
        ax.set_xscale("log")
    else:
        edges = np.linspace(allv.min(), allv.max(), bins)
    for k in (0, 1):
        v = data[k].clip(*clip) if clip is not None else data[k]
        if log:
            v = v.clip(lower=edges[0])
        ax.hist(v, bins=edges, density=True, histtype="step", lw=1.6, color=CLASS_COLOR[k])
        ax.hist(v, bins=edges, density=True, histtype="stepfilled", alpha=0.12, color=CLASS_COLOR[k])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)


def fig_distributions(df, out):
    panels = [
        ("amount", "Voucher amount", "Rand", False, None),
        ("amount_ratio_to_avg_90d", "Amount vs own 90-day average", "ratio (log scale)", True, None),
        ("amount_to_limit_ratio", "Amount vs daily limit", "share of limit", False, (0, 1)),
        ("secs_since_login", "Time from login to voucher", "seconds (log scale)", True, None),
        ("redeem_delay_min", "Time until cash-out", "minutes (log scale)", True, None),
        ("redeem_dist_usual_km", "Cash-out distance from usual area", "km (log scale)", True, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    for ax, (col, title, xl, log, clip) in zip(axes.flat, panels):
        density_panel(ax, df, col, title, xl, log=log, clip=clip)
    axes.flat[0].set_ylabel("Density (each class sums to 1)")
    axes.flat[3].set_ylabel("Density (each class sums to 1)")
    class_legend(axes.flat[0], loc="upper right")
    fig.tight_layout()
    save(fig, out, "feature_distributions")
    # individual versions for flexible placement in the report
    for col, title, xl, log, clip in panels:
        f, ax = plt.subplots(figsize=(3.6, 2.6))
        density_panel(ax, df, col, title, xl, log=log, clip=clip)
        ax.set_ylabel("Density")
        class_legend(ax, loc="upper right")
        save(f, out, f"dist_{col}")


def fig_flags(df, out):
    flags = {
        "recipient_known": "Recipient used before",
        "recipient_added_in_session": "Recipient added in session",
        "is_self_send": "Sent to own number",
        "limit_changed_24h": "Limit raised in last 24h",
        "new_device_7d": "New device in last 7 days",
        "vouchers_in_session": "Not first voucher in session",
        "has_history": "Has earlier vouchers",
        "is_night": "Created at night (21:00-06:00)",
    }
    rates = pd.DataFrame({k: [(df.loc[df.is_scam == c, f] > 0).mean() for c in (0, 1)] for f, k in flags.items()},
                         index=[0, 1]).T
    rates = rates.sort_values(1)
    y = np.arange(len(rates))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    h = 0.36
    for k, off in ((0, -h / 2), (1, h / 2)):
        ax.barh(y + off, rates[k] * 100, height=h, color=CLASS_COLOR[k], edgecolor="white", linewidth=1)
        for yy, v in zip(y + off, rates[k] * 100):
            ax.text(v + 1, yy, f"{v:.0f}%", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, rates.index)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Share of vouchers where the flag is present (%)")
    ax.set_title("Binary indicators by class")
    ax.grid(axis="y", visible=False)
    class_legend(ax, loc="lower right")
    save(fig, out, "flag_rates")


def fig_hour(df, out):
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    bins = np.arange(0, 25)
    for k in (0, 1):
        v = df.loc[df.is_scam == k, "hour"]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=CLASS_COLOR[k])
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.12, color=CLASS_COLOR[k])
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of voucher creation")
    ax.set_yticks([])
    ax.set_title("Time of day by class")
    ax.grid(axis="y", visible=False)
    class_legend(ax, loc="upper left")
    save(fig, out, "hour_of_day")


def fig_correlation(df, out):
    cols = ["amount", "amount_ratio_to_avg_90d", "amount_to_limit_ratio", "history_count",
            "days_since_last_voucher", "vouchers_24h", "vouchers_in_session", "recipient_known",
            "is_self_send", "recipient_added_in_session", "secs_since_login", "screens_before_create",
            "hour_deviation", "limit_changed_24h", "new_device_7d", "account_age_days",
            "redeem_delay_min", "redeem_dist_usual_km", "is_scam"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(cols)), cols, fontsize=7.5)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if i != j and abs(v) >= 0.3:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Spearman correlation", color=INK2)
    ax.set_title("Feature correlations (Spearman); |r| >= 0.3 labelled")
    save(fig, out, "correlation_heatmap")
    return corr["is_scam"].drop("is_scam").sort_values(key=abs, ascending=False)


def fig_benford(df, out):
    first = df["amount"].astype(int).astype(str).str[0].astype(int)
    observed = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    expected = pd.Series(np.log10(1 + 1 / np.arange(1, 10)), index=range(1, 10))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.bar(observed.index, observed * 100, color=LEGIT, width=0.6, edgecolor="white", linewidth=1,
           label="Generated amounts")
    ax.plot(expected.index, expected * 100, color=INK2, lw=2, marker="o", ms=4, label="Benford expected")
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("First digit of amount")
    ax.set_ylabel("Share (%)")
    ax.set_title("First-digit distribution vs Benford's Law")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, out, "benford")
    mad = (observed - expected).abs().mean()
    return pd.DataFrame({"observed": observed, "expected": expected}), mad


def summary_table(df, out):
    feats = ["amount", "amount_ratio_to_avg_90d", "amount_to_limit_ratio", "history_count",
             "secs_since_login", "screens_before_create", "redeem_delay_min", "redeem_dist_usual_km"]
    rows = []
    for f in feats:
        r = {"feature": f}
        for k in (0, 1):
            v = df.loc[df.is_scam == k, f].dropna()
            r[f"{CLASS_LABEL[k]} median"] = v.median()
            r[f"{CLASS_LABEL[k]} IQR"] = f"{v.quantile(.25):.1f}-{v.quantile(.75):.1f}"
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "tables", "feature_summary_by_class.csv"), index=False)
    t.to_latex(os.path.join(out, "tables", "feature_summary_by_class.tex"), index=False, float_format="%.1f")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df = pd.read_csv(f"{a.data}/cardless_features.csv")
    latent = pd.read_csv(f"{a.data}/customers_latent.csv")

    fig_scenarios(df, a.out)
    fig_activity(df, latent, a.out)
    fig_distributions(df, a.out)
    fig_flags(df, a.out)
    fig_hour(df, a.out)
    corr = fig_correlation(df, a.out)
    benford, mad = fig_benford(df, a.out)
    t = summary_table(df, a.out)

    print("Spearman correlation with is_scam (top 10):")
    print(corr.head(10).round(3).to_string())
    print(f"\nBenford mean absolute deviation: {mad:.4f}")
    print(benford.round(3).T.to_string())
    print("\n" + t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
