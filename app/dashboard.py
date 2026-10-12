"""
Cardless withdrawal risk monitoring dashboard (prototype).

Replays the test period (days 120-179) as a bank analyst would see it: the alert queue for a chosen day,
each alert's risk band and reasons, a drill-down into the customer's recent behaviour, and how the model
performs. Scores come from the Rules + One-Class SVM hybrid (src/score_all.py), which was fitted and tuned
only on earlier days.

Run:
    python src/score_all.py          # once, after generating the dataset
    streamlit run app/dashboard.py
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "synthetic")
REPORT = os.path.join(ROOT, "report")
START = pd.Timestamp("2025-07-01")

BANDS = ["Low", "Medium", "High", "Critical"]
# status palette (dataviz reference): never reused for series, always shown with a label
BAND_COLOR = {"Low": "#898781", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
BAND_ICON = {"Low": "○", "Medium": "◐", "High": "●", "Critical": "▲"}
LEGIT, SCAM = "#2a78d6", "#eb6834"
EVENT_LABEL = {"login": "Login", "add_recipient": "Recipient added", "change_limit": "Daily limit changed",
               "device_registered": "New device registered", "password_reset": "Password reset",
               "create_voucher": "Voucher created", "redeem_voucher": "Voucher cashed out", "logout": "Logout"}

st.set_page_config(page_title="Cardless Risk Monitor", page_icon="🛡️", layout="wide")


@st.cache_data
def load():
    v = pd.read_csv(os.path.join(DATA, "scored_vouchers.csv"), parse_dates=["created_at", "redeemed_at"])
    cust = pd.read_csv(os.path.join(DATA, "customers.csv")).set_index("customer_id")
    with open(os.path.join(DATA, "dashboard_meta.json")) as f:
        meta = json.load(f)
    return v, cust, meta


@st.cache_data
def customer_events(customer_id):
    ev = pd.read_parquet(os.path.join(DATA, "customer_events_test.parquet"),
                         filters=[("customer_id", "==", customer_id)])
    ev["time"] = START + pd.to_timedelta(ev.t_min, unit="min")
    return ev


def band_label(b):
    return f"{BAND_ICON[b]} {b}"


def band_scale():
    return alt.Scale(domain=BANDS, range=[BAND_COLOR[b] for b in BANDS])


if not os.path.exists(os.path.join(DATA, "scored_vouchers.csv")):
    st.error("No scores found. Run `python src/simulate.py`, `python src/features.py` and "
             "`python src/score_all.py` first.")
    st.stop()

v, customers, meta = load()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ Cardless Risk Monitor")
stage_name = st.sidebar.radio("Scoring point", ["A: at voucher creation", "B: at cash-out"],
                              help="Stage A scores when the voucher is created. Stage B re-scores it when "
                                   "someone tries to cash it out, adding where and how fast that happens.")
stage = stage_name[0].lower()
truth = st.sidebar.toggle("Show ground truth (evaluation mode)", value=True,
                          help="A real analyst would not see this. It shows which vouchers were scams in the "
                               "simulation, so the alerts can be judged.")
days = sorted(v.day.unique())
day = st.sidebar.select_slider("Day", options=days, value=days[0],
                               format_func=lambda d: (START + pd.Timedelta(days=int(d))).strftime("%a %d %b %Y"))
cut = meta["stages"][stage.upper()]["cutoffs"]
st.sidebar.caption(
    f"Model: {meta['model']} (w = {meta['stages'][stage.upper()]['hybrid_w']}). "
    f"Bands on the 0–100 risk score: Medium ≥ {cut['Medium']:.1f}, High ≥ {cut['High']:.1f}, "
    f"Critical ≥ {cut['Critical']:.1f}. High and Critical raise an alert. Fitted on days 0–89, tuned on "
    "days 90–119, never on the days shown here.")

scored = v if stage == "a" else v[v.redeemed == 1]
risk, band, alert, reasons = f"risk_{stage}", f"band_{stage}", f"alert_{stage}", f"reasons_{stage}"
scored = scored.assign(**{alert: scored[alert].astype(bool)})
scored["caught"] = scored.is_scam.astype(bool) & scored[alert]
today = scored[scored.day == day]

tab_queue, tab_customer, tab_model = st.tabs(["Alert queue", "Customer drill-down", "Model performance"])

# ---------------------------------------------------------------- alert queue
with tab_queue:
    alerts = today[today[alert]]
    c = st.columns(4 if truth else 3)
    c[0].metric("Vouchers scored", f"{len(today):,}")
    c[1].metric("Alerts (High + Critical)", f"{len(alerts):,}", f"{len(alerts) / max(len(today), 1):.1%} of vouchers",
                delta_color="off")
    c[2].metric("Amount under alert", f"R{alerts.amount.sum():,.0f}")
    if truth:
        caught, scams = int(alerts.is_scam.sum()), int(today.is_scam.sum())
        c[3].metric("Scams caught", f"{caught} of {scams}",
                    f"{caught / len(alerts):.0%} of alerts are scams" if len(alerts) else "no alerts",
                    delta_color="off")

    # daily trend over the replay window
    daily = scored.groupby("day").agg(alerts=(alert, "sum"), scams=("is_scam", "sum"), caught=("caught", "sum"))
    daily["date"] = START + pd.to_timedelta(daily.index, unit="D")
    series = {"Alerts raised": "alerts"} | ({"Scams caught": "caught"} if truth else {})
    long = daily.reset_index().melt(id_vars=["day", "date"], value_vars=list(series.values()))
    long["series"] = long.variable.map({v_: k for k, v_ in series.items()})
    base = alt.Chart(long).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title="Vouchers per day"),
        color=alt.Color("series:N", scale=alt.Scale(domain=list(series), range=[LEGIT, SCAM][:len(series)]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("date:T", format="%a %d %b"), "series:N", alt.Tooltip("value:Q", title="vouchers")])
    rule = alt.Chart(pd.DataFrame({"date": [START + pd.Timedelta(days=int(day))]})).mark_rule(
        color="#52514e", strokeDash=[3, 3]).encode(x="date:T")
    st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(size=30, filled=True, opacity=0) + rule)
                    .properties(height=180).interactive(bind_y=False), use_container_width=True)

    st.subheader("Queue")
    show = st.multiselect("Bands", BANDS, default=["Critical", "High"], format_func=band_label)
    q = today[today[band].isin(show)].sort_values(risk, ascending=False)
    time_col = "created_at" if stage == "a" else "redeemed_at"
    table = pd.DataFrame({
        "Time": q[time_col].dt.strftime("%H:%M"),
        "Risk": q[risk].round(1),
        "Band": q[band].map(band_label),
        "Voucher": q.voucher_id.astype(str),
        "Customer": q.customer_id.astype(str),
        "Amount (R)": q.amount.astype(int),
    })
    if truth:
        table["Truth"] = q.scenario_type.str.replace("_", " ")
    table["Why it was flagged"] = q[reasons].fillna("").str.replace(" | ", " • ", regex=False)
    st.dataframe(table, hide_index=True, use_container_width=True, height=420,
                 column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                                        format="%.1f", width="small"),
                                "Why it was flagged": st.column_config.TextColumn(width="large")})
    st.caption("Suggested handling: Critical → hold the voucher and call the customer back; High → in-app warning "
               "and a short cooling-off delay; Medium → monitor. Most legitimate emergencies land in High, so a "
               "step-up check suits them better than a block.")

# ---------------------------------------------------------------- customer drill-down
with tab_customer:
    ranked = today.sort_values(risk, ascending=False).customer_id.drop_duplicates().tolist()
    cid = st.selectbox("Customer (today's highest risk first)", ranked,
                       format_func=lambda c_: f"{c_} — top risk today "
                                              f"{today.loc[today.customer_id == c_, risk].max():.1f}")
    prof = customers.loc[cid]
    hist = v[v.customer_id == cid].sort_values("created_at")
    c = st.columns(4)
    c[0].metric("Home", prof.home_metro)
    c[1].metric("Account opened", str(prof.account_open_date))
    c[2].metric("Daily limit at start", f"R{prof.daily_limit_at_start:,}")
    c[3].metric("Vouchers in test period", len(hist))

    hb = hist.assign(Band=hist[f"band_{stage}"].fillna("Low"), risk=hist[f"risk_{stage}"].round(1),
                     truth=hist.scenario_type.str.replace("_", " "))
    tips = [alt.Tooltip("created_at:T", title="created", format="%d %b %H:%M"), alt.Tooltip("amount:Q", format=",.0f"),
            "Band:N", alt.Tooltip("risk:Q", title="risk")] + ([alt.Tooltip("truth:N")] if truth else [])
    dots = alt.Chart(hb).mark_circle(size=110, stroke="#fcfcfb", strokeWidth=2, opacity=1).encode(
        x=alt.X("created_at:T", title=None),
        y=alt.Y("amount:Q", title="Voucher amount (R)"),
        color=alt.Color("Band:N", scale=band_scale(), legend=alt.Legend(orient="top", title="Risk band")),
        tooltip=tips)
    layers = dots
    if truth and hb.is_scam.any():
        layers = layers + alt.Chart(hb[hb.is_scam == 1]).mark_point(shape="diamond", size=320, color=SCAM,
                                                                     strokeWidth=1.5).encode(
            x="created_at:T", y="amount:Q", tooltip=tips)
    st.altair_chart(layers.properties(height=230), use_container_width=True)
    if truth:
        st.caption("Orange diamond = scam voucher in the simulation.")

    ev = customer_events(cid)
    window = st.slider("Event window (days before the selected day)", 1, 30, 7)
    t1 = START + pd.Timedelta(days=int(day) + 1)
    recent = ev[(ev.time >= t1 - pd.Timedelta(days=window)) & (ev.time < t1) & ev.event_type.isin(
        [e for e in EVENT_LABEL if e not in ("login", "logout")])]
    st.subheader("Recent account activity")
    st.dataframe(pd.DataFrame({
        "Time": recent.time.dt.strftime("%d %b %H:%M"),
        "Event": recent.event_type.map(EVENT_LABEL),
        "Session": recent.session_id,
        "Device": recent.device_id,
        "Amount (R)": recent.amount.fillna(0).astype(int).replace(0, pd.NA),
        "Recipient": recent.recipient_id,
        "New limit (R)": recent.new_limit,
        "Channel": recent.channel,
    }).iloc[::-1], hide_index=True, use_container_width=True, height=300)

    st.subheader("Vouchers")
    st.dataframe(pd.DataFrame({
        "Created": hist.created_at.dt.strftime("%d %b %H:%M"),
        "Amount (R)": hist.amount.astype(int),
        "Risk A": hist.risk_a.round(1), "Band A": hist.band_a.map(band_label),
        "Risk B": hist.risk_b.round(1), "Band B": hist.band_b.map(band_label, na_action="ignore"),
        "Reasons (stage B if cashed out)": hist.reasons_b.fillna(hist.reasons_a).fillna("")
        .str.replace(" | ", " • ", regex=False),
    } | ({"Truth": hist.scenario_type.str.replace("_", " ")} if truth else {})).iloc[::-1],
        hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- model performance
with tab_model:
    st.markdown("Performance over the whole test period (days 120–179). Everything was tuned on earlier days.")
    res = pd.read_csv(os.path.join(REPORT, "tables", "model_comparison.csv"))
    res = res[res["operating point"].str.startswith("max F1")]
    res = res[res.stage == stage.upper()][["model", "precision", "recall", "f1", "fpr", "pr_auc", "roc_auc"]]
    res.columns = ["Model", "Precision", "Recall", "F1", "False positive rate", "PR-AUC", "ROC-AUC"]
    st.dataframe(res.style.format(precision=3).highlight_max(subset=["F1", "PR-AUC"], color="#fde7dd"),
                 hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Risk bands by class (whole test period)**")
        tab = pd.crosstab(scored[band], scored.is_scam).reindex(BANDS).fillna(0).astype(int)
        tab.columns = ["Legitimate", "Scam"]
        tab["Share of scams"] = (tab.Scam / tab.Scam.sum()).map("{:.0%}".format)
        tab["Precision"] = (tab.Scam / (tab.Scam + tab.Legitimate)).map("{:.0%}".format)
        tab.index = [band_label(b) for b in tab.index]
        st.dataframe(tab, use_container_width=True)
    with right:
        st.markdown("**Detection by scenario (alerts = High + Critical)**")
        sc = scored.groupby("scenario_type")[alert].agg(["mean", "size"]).reset_index()
        sc["kind"] = sc.scenario_type.str.startswith("scam").map({True: "Scam", False: "Legitimate"})
        sc["label"] = sc.scenario_type.str.replace("_", " ")
        order = [s_.replace("_", " ") for s_ in ["scam_classic", "scam_self_send", "scam_limit_raise", "scam_split",
                                                  "scam_subtle", "normal", "legit_emergency", "legit_new_phone",
                                                  "legit_travel"]]
        bars = alt.Chart(sc).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("label:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.25),
                    axis=alt.Axis(labelOverlap=False, labelLimit=160)),
            x=alt.X("mean:Q", title="Share of vouchers flagged", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Legitimate", "Scam"], range=[LEGIT, SCAM]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["label:N", alt.Tooltip("mean:Q", format=".1%", title="flagged"),
                     alt.Tooltip("size:Q", title="vouchers")])
        st.altair_chart(bars.properties(height=280), use_container_width=True)

    for name, caption in [("pr_curves", "Precision–recall curves for every model"),
                          ("learning_curve", "Learning curve: performance against training size"),
                          ("sensitivity", "Sensitivity: rarer scams, rule ablation, rule-weight perturbation")]:
        path = os.path.join(REPORT, "figures", f"{name}.png")
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
