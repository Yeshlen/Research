"""Start the multipage prototype with: streamlit run app/dashboard.py"""
import streamlit as st
from utils import MODEL_NAMES, MODEL_NOTES, load_resources, model_label, page_setup

page_setup("Home")
bundle, results, vouchers = load_resources()
st.title("Cardless Risk Monitor")
st.write("Explore the scoring methods, test an existing voucher, or generate an independent simulated example.")
cols = st.columns(3)
cols[0].metric("Research vouchers", f"{len(vouchers):,}")
cols[1].metric("Active customers", f"{vouchers.customer_id.nunique():,}")
cols[2].metric("Scam vouchers", f"{vouchers.is_scam.mean():.2%}")
st.header("Scoring methods")
for name in MODEL_NAMES:
    st.write(f"**{model_label(name)}:** {MODEL_NOTES[name]}")
st.header("Generate a record")
st.write("Use the Generate Record page to generate legitimate or scam activity. Random makes a 50/50 choice between the two. Each example starts from a separate copy of an existing customer's history. Generated examples stay within your current session.")
st.write("The model can miss a scam or flag a legitimate voucher. The app shows both its decision and the known simulation label.")
st.caption("Research data, trained models and published test metrics remain unchanged. This prototype does not connect to a bank or block payments.")
