"""Generate isolated examples, persist them separately, and score with fixed models."""
import pandas as pd
import streamlit as st
from utils import MODEL_NAMES, ROOT, alert_dialog, generation_start, load_resources, model_label, page_setup, show_result
from demo_scoring import score_vouchers
from demo_simulation import (LEGIT_SCENARIOS, SCAM_SCENARIOS, generate_example, load_examples, save_example)

page_setup("Generate Record")
bundle, _, _ = load_resources()
st.title("Generate Record")
st.write("Generate new activity for an existing customer, then apply a saved scoring method.")
model_name = st.selectbox("Scoring method", MODEL_NAMES, index=4, format_func=model_label, help="Choose the method that scores the new record.")
requested = st.radio("Activity to generate", ["Legitimate", "Scam", "Random (50/50)"],
                     index=None, horizontal=True, key="generation_kind", help="Choose the activity type. Random gives each type an equal chance.")
scenario = "Automatic"
if requested == "Legitimate":
    scenario = st.selectbox("Legitimate scenario", LEGIT_SCENARIOS, help="Choose the legitimate activity to generate.",
                            format_func=lambda x: x.replace("legit_", "").replace("_", " ").capitalize())
elif requested == "Scam":
    scenario = st.selectbox("Scam scenario", ["Automatic"] + SCAM_SCENARIOS, help="Choose a scam type, or let the generator choose.",
                            format_func=lambda x: x.replace("scam_", "").replace("_", " ").capitalize())
    st.caption("Automatic uses the original scam-scenario weights.")
elif requested == "Random (50/50)":
    st.caption("An equal chance of a scam or an ordinary legitimate session. Scam types follow the original scenario weights.")
generate_clicked = st.button("Generate", type="primary", disabled=requested is None, help="Create and score new activity using an existing customer history.")

generation_succeeded = False
if generate_clicked:
    # Class is chosen once. Generation is never retried based on a model's decision.
    import numpy as np
    actual_kind = str(np.random.default_rng().choice(["Legitimate", "Scam"])) if requested == "Random (50/50)" else requested
    try:
        with st.spinner("Copying an earlier customer history and generating new events…"):
            features, events = generate_example(ROOT, actual_kind, scenario)
            save_example(ROOT, features, events)
        decisions = score_vouchers(features, bundle["stages"]["A"], model_name, "A")
        st.session_state["simulation_result"] = dict(features=features, events=events,
                                                   decisions=decisions, stage="A", model=model_name)
        generation_succeeded = True
    except (ValueError, OSError) as exc:
        st.error(str(exc))

latest = st.session_state.get("simulation_result")
redemption_clicked = False
if latest:
    features = latest["features"]
    st.subheader("Latest generated example")
    st.caption(f"Customer history used up to {features.iloc[0].history_cutoff}. This example has its own new events.")
    if latest["stage"] == "A":
        eligible = features[features.redeemed == 1]
        if not eligible.empty and st.button("Simulate redemption", type="primary", help="Score this example using its redemption information."):
            latest["decisions"] = score_vouchers(eligible, bundle["stages"]["B"], latest["model"], "B")
            latest["stage"] = "B"
            st.session_state["simulation_result"] = latest
            redemption_clicked = True
        elif eligible.empty:
            st.info("No redemption was generated within the observation window. This example can only be scored at creation.")
    decisions = latest["decisions"]
    table = pd.DataFrame(decisions)[["voucher_id", "amount", "risk", "threshold", "alert"]].rename(
        columns={"voucher_id": "Voucher", "amount": "Amount (R)", "risk": "Risk score",
                 "threshold": "Alert threshold", "alert": "Alert"})
    st.dataframe(table.style.format({"Amount (R)": "{:,.0f}", "Risk score": "{:.2f}",
                                     "Alert threshold": "{:.2f}"}), hide_index=True, width="stretch")
    st.caption(f"Scored with {model_label(latest['model'])} at {'creation' if latest['stage'] == 'A' else 'redemption'}. Split sessions may contain several vouchers.")
    chosen = st.selectbox("Inspect generated voucher", list(range(len(decisions))),
                          format_func=lambda i: str(decisions[i]["voucher_id"]),
                          key=f"inspect_{features.iloc[0].simulation_id}_{latest['stage']}", help="Choose a voucher from the latest generated example.")
    decision = decisions[chosen]
    record = features[features.voucher_id == decision["voucher_id"]]
    show_result(decision, record, "Generated examples")
    with st.expander("Generated banking events"):
        events = latest["events"].copy()
        if latest["stage"] == "A":
            events = events[events.event_type != "redeem_voucher"]
        # Account events have blank voucher/amount cells; display them as missing numbers.
        events["voucher_id"] = pd.to_numeric(events.voucher_id, errors="coerce").astype("Int64")
        events["amount"] = pd.to_numeric(events.amount, errors="coerce")
        events["Event time"] = pd.Timestamp(generation_start(bundle)) + pd.to_timedelta(events.t_min, unit="min")
        event_view = events[["Event time", "event_type", "voucher_id", "recipient_id", "amount"]].rename(
            columns={"event_type": "Event", "voucher_id": "Voucher", "recipient_id": "Recipient", "amount": "Amount (R)"})
        st.dataframe(event_view, hide_index=True, width="stretch")
    alerts = [d for d in decisions if d["alert"]]
    if (generation_succeeded or redemption_clicked) and alerts:
        alert_dialog(alerts)

st.divider()
examples = load_examples(ROOT)
if not examples.empty:
    st.subheader("Saved generated vouchers")
    display = examples[["simulation_id", "voucher_id", "customer_id", "created_at", "amount", "scenario_type", "is_scam"]].copy()
    display["scenario_type"] = display.scenario_type.str.replace("_", " ")
    display["is_scam"] = display.is_scam.map({1: "Scam", 0: "Legitimate"})
    display = display.rename(columns={"simulation_id": "Example", "voucher_id": "Voucher", "customer_id": "Customer",
                                      "created_at": "Created at", "amount": "Amount (R)",
                                      "scenario_type": "Scenario", "is_scam": "Simulation label"})
    st.dataframe(display, hide_index=True, width="stretch")
    st.download_button("Download generated voucher CSV", examples.to_csv(index=False).encode(),
                       file_name="generated_vouchers.csv", mime="text/csv", help="Download only the separately generated vouchers.")
st.caption("Generated examples are saved separately. They never update the original customer history, research dataset or test-performance tables.")
