"""Shared loading, record selection and display functions for the Streamlit pages."""
from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core_evaluation import fingerprint, load_data, split_stage
from demo_scoring import score_vouchers
from deployment_data import ensure_dataset, ensure_runtime
from session_examples import session_examples
from models import RULES, STAGE_A_FEATURES, STAGE_B_FEATURES

MODEL_NAMES = ["Rules", "Isolation Forest", "LOF", "One-Class SVM", "Hybrid"]
MODEL_NOTES = {
    "Rules": "A checklist adds points for warning indicators. The total is capped at 100 and compared with the saved alert threshold.",
    "Isolation Forest": "Random partitions isolate unusual feature patterns. Vouchers that are easier to isolate receive higher anomaly scores.",
    "LOF": "Local Outlier Factor compares a voucher with nearby observations. A voucher in a less dense area than its neighbours receives a higher anomaly score.",
    "One-Class SVM": "One-Class SVM learns a boundary from the fitting data. Vouchers that fall outside its usual pattern can receive higher anomaly scores.",
    "Hybrid": "The hybrid gives equal weight to the rule score and the validation-selected anomaly model. Both components use a 0–1 scale before blending.",
}


def model_label(name):
    return "Rule Based" if name == "Rules" else name


def page_setup(title):
    st.set_page_config(page_title=f"{model_label(title)} | Cardless Risk Monitor", layout="wide")
    st.sidebar.title("Cardless Risk Monitor")


def stage_control():
    return st.sidebar.radio("Scoring point", ["A", "B"],
                            format_func=lambda x: "Creation" if x == "A" else "Redemption",
                            key="scoring_stage", help="Choose when the voucher is scored.")


@st.cache_resource
def _load_resources(signatures):
    bundle = joblib.load(ROOT / "report/core/models/fitted_models.joblib")
    if bundle["fingerprint"] != fingerprint(ROOT):
        raise ValueError("The data or fitting settings changed. Run notebooks 02 and 03 again before using the dashboard.")
    results = pd.read_csv(ROOT / "report/core/tables/model_comparison.csv")
    vouchers = load_data(ROOT)
    return bundle, results, vouchers


@st.cache_resource
def _prepare_dataset():
    ensure_dataset(ROOT)
    ensure_runtime(ROOT)


def load_resources():
    try:
        with st.spinner("Preparing the demonstration data…"):
            _prepare_dataset()
    except (OSError, ValueError) as exc:
        st.error(f"The demonstration data could not be prepared: {exc}")
        st.info("The dataset release must be publicly accessible. Reload the page to retry.")
        st.stop()
    paths = [ROOT / "report/core/models/fitted_models.joblib",
             ROOT / "report/core/tables/model_comparison.csv", ROOT / "data/synthetic/cardless_features.csv",
             ROOT / "config/generator.yaml", ROOT / "src/models.py", ROOT / "src/core_evaluation.py",
             ROOT / "notebooks/02_fit_models.ipynb", ROOT / "data/synthetic/dataset_metadata.json"]
    required = paths[:3]
    if not all(p.exists() for p in required):
        st.info("Run notebooks 01, 02 and 03 to prepare the dataset, saved models and results.")
        st.stop()
    signatures = tuple((str(p), p.stat().st_mtime_ns) for p in paths if p.exists())
    try:
        return _load_resources(signatures)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()


def example_label(row):
    scenario = str(row.get("scenario_type", "example")).replace("_", " ")
    return f"Voucher {int(row.voucher_id)} · Customer {int(row.customer_id)} · {scenario}"


def select_record(vouchers, stage, start_date):
    source = st.radio("Record source", ["Existing test records", "Generated examples"],
                      horizontal=True, key="record_source", help="Use a test voucher or a saved generated example.")
    if source == "Existing test records":
        records = split_stage(vouchers, stage, start_date)[2]
        label_filter = st.selectbox("Browse simulation labels", ["All", "Scam", "Legitimate"], help="Filter by the known simulation label.")
        if label_filter != "All":
            records = records[records.is_scam == int(label_filter == "Scam")]
        if records.empty:
            st.info("No eligible test vouchers for this selection.")
            return None, source
        record_key = f"test_number_{stage}_{label_filter}"
        number_col, random_col = st.columns([8, 1], vertical_alignment="bottom")
        with random_col:
            if st.button("", icon=":material/casino:", key=f"random_record_{stage}_{label_filter}",
                         help="Pick a random voucher from the current filter."):
                # Choose a different record when more than one is available.
                current = int(st.session_state.get(record_key, 1))
                count = len(records)
                choice = int(np.random.default_rng().integers(1, count)) if count > 1 else 1
                st.session_state[record_key] = choice + (choice >= current) if count > 1 else 1
        with number_col:
            number = st.number_input("Test record number", min_value=1, max_value=len(records), step=1,
                                     key=record_key, help="Choose a voucher from the filtered test records.")
        record = records.iloc[[int(number) - 1]].copy()
        st.caption(f"{len(records):,} eligible test vouchers. Changing this selection does not change the reported test metrics.")
    else:
        records = session_examples(st.session_state)
        if records.empty:
            st.info("Create an example on the Generate Record page first.")
            return None, source
        if stage == "B":
            records = records[records.redeemed == 1]
        if records.empty:
            st.info("No generated voucher has a simulated redemption in the observation window. Use Creation or generate another example.")
            return None, source
        records = records.reset_index(drop=True)
        chosen = st.selectbox("Generated voucher", list(range(len(records))),
                              format_func=lambda i: example_label(records.iloc[i]),
                              index=len(records) - 1, key=f"generated_choice_{stage}", help="Choose a previously generated voucher.")
        record = records.iloc[[chosen]].copy()
        st.caption("Generated examples belong to your current session and do not change customer histories or test results.")
    row = record.iloc[0]
    st.write(f"**Voucher {int(row.voucher_id)}** · Customer {int(row.customer_id)} · **R{row.amount:,.0f}**")
    st.caption(f"Created at {row.created_at}")
    if source == "Generated examples" and "history_cutoff" in row:
        st.caption(f"Independent history checkpoint: {row.history_cutoff}")
    with st.expander("View scoring features"):
        fields = STAGE_A_FEATURES if stage == "A" else STAGE_B_FEATURES
        st.dataframe(record[fields].T.rename(columns={record.index[0]: "Value"}), width="stretch")
    return record, source


def show_result(decision, record, source):
    if decision["alert"]:
        st.error("Alert: this voucher crossed the selected method's threshold.")
    else:
        st.success("No alert: this voucher did not cross the selected method's threshold.")
    cols = st.columns(2)
    cols[0].metric("Risk score (0–100)", f"{decision['risk']:.2f}")
    cols[1].metric("Alert threshold", f"{decision['threshold']:.2f}")
    st.caption("Higher means more suspicious. The risk score is not a probability of a scam.")
    if decision["model"] == "Hybrid":
        cols = st.columns(2)
        cols[0].metric("Rule score", f"{decision['rule_score']:.0f} / 100")
        cols[1].metric("Anomaly percentile", f"{100 * decision['anomaly_percentile']:.2f}")
        st.caption(f"Hybrid = 0.5 × (rule score / 100) + 0.5 × {decision['anomaly_model']} anomaly percentile.")
    st.subheader("Rule indicators present")
    if decision["reasons"]:
        for reason in decision["reasons"]:
            st.write(f"- {reason}")
    else:
        st.write("No checklist conditions were triggered.")
    if decision["model"] != "Rules":
        st.caption("These are checklist indicators. They do not explain the internal anomaly-model decision.")
    actual = int(record.iloc[0].is_scam)
    st.write(f"**Simulation label:** {'Scam' if actual else 'Legitimate'}")
    if decision["alert"] and not actual:
        st.warning("False alert: this legitimate voucher was flagged.")
    elif not decision["alert"] and actual:
        st.warning("Missed scam: this scam voucher was not flagged.")
    else:
        st.caption("The alert decision agrees with the simulation label.")


@st.dialog("Suspicious voucher detected")
def alert_dialog(decisions):
    st.write("The following vouchers crossed the selected method's saved alert threshold.")
    for decision in decisions:
        st.write(f"**Voucher {decision['voucher_id']} · R{decision['amount']:,.0f}**")
        st.write(f"{model_label(decision['model'])} · {'Creation' if decision['stage'] == 'A' else 'Redemption'}")
        st.write(f"Score **{decision['risk']:.2f}** / 100 · threshold **{decision['threshold']:.2f}**")
        for reason in decision["reasons"]:
            st.write(f"- {reason}")
        if not decision["reasons"]:
            st.write("No checklist rules triggered; the anomaly score crossed the threshold.")
    if any(d["model"] != "Rules" for d in decisions):
        st.caption("Checklist indicators provide context; they are not feature attributions for an anomaly model.")
    st.caption("This is a review alert, not confirmation of fraud. No payment is blocked.")
    if st.button("Close", key="close_alert"):
        st.rerun()


def show_performance(results, model_name, stage, selected):
    row = results[(results.stage == stage) & (results.model == model_name)].iloc[0]
    st.subheader("Saved test performance")
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Precision", "Recall", "F1", "False positive rate"],
                                 [f"{row.precision:.1%}", f"{row.recall:.1%}", f"{row.f1:.3f}", f"{row.fpr:.2%}"]):
        col.metric(label, value)
    matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No alert", "Alert"],
           yticklabels=["Legitimate", "Scam"], xlabel="System decision", ylabel="Simulation label",
           title=f"{model_label(model_name)}: {'creation' if stage == 'A' else 'redemption'}")
    fig.tight_layout()
    st.pyplot(fig, width="content")
    plt.close(fig)
    st.caption("Precision: how many alerts were scams. Recall: how many scams were detected. F1 balances both.")

    if model_name == "Rules":
        st.subheader("Rules and points")
        labels = {
            "multi_in_session": "Second or later voucher in the same session",
            "velocity_24h": "Third or later voucher within 24 hours",
            "long_session": "At least 5 minutes since login or 6 screens viewed",
            "fast_redemption": "Redeemed in under 15 minutes",
        }
        st.table(pd.DataFrame([
            {"Rule": labels.get(name, reason[0].upper() + reason[1:]),
             "Points": points, "Available from": "Creation" if rule_stage == "A" else "Redemption"}
            for name, points, rule_stage, _, reason in RULES
        ]))
        st.caption("Matching rules add points, capped at 100. Creation rules also apply at redemption. Points were manually chosen.")
    elif model_name == "Hybrid":
        st.subheader("Hybrid formula")
        st.latex(r"H = 0.5 \times \frac{R}{100} + 0.5 \times A")
        st.write(f"R is the Rule Based score (0–100). A is the {selected['hybrid_anomaly']} anomaly percentile (0–1).")
        st.caption("The dashboard displays H × 100 as the risk score.")


def show_model_page(model_name):
    page_setup(model_name)
    stage = stage_control()
    bundle, results, vouchers = load_resources()
    selected = bundle["stages"][stage]
    st.title(model_label(model_name))
    st.write(MODEL_NOTES[model_name])
    if model_name == "Hybrid":
        st.caption(f"Saved anomaly component at this stage: {selected['hybrid_anomaly']}.")
    testing, performance = st.tabs(["Test a voucher", "Performance"])
    with testing:
        record, source = select_record(vouchers, stage, generation_start(bundle))
        if record is not None:
            key = (model_name, stage, source, int(record.iloc[0].voucher_id), bundle["fingerprint"])
            clicked = st.button("Score voucher", type="primary", help="Apply this method to the selected voucher.")
            if clicked:
                decision = score_vouchers(record, selected, model_name, stage)[0]
                st.session_state["latest_prediction"] = (key, decision)
            previous = st.session_state.get("latest_prediction")
            if previous and previous[0] == key:
                show_result(previous[1], record, source)
                if clicked and previous[1]["alert"]:
                    alert_dialog([previous[1]])
            if source == "Generated examples" and stage == "A" and int(record.iloc[0].redeemed) == 1:
                st.button("Simulate redemption", help="Switch to scoring with redemption information.", on_click=lambda: st.session_state.update(scoring_stage="B"))
    with performance:
        show_performance(results, model_name, stage, selected)


def generation_start(bundle):
    # The generation metadata is authoritative for the existing dataset's start date.
    path = ROOT / "data/synthetic/dataset_metadata.json"
    if path.exists():
        return json.loads(path.read_text())["generation_config"]["start_date"]
    return "2025-07-01"
