"""Historical replay prototype showing the notebook-generated core evaluation."""
from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from core_evaluation import fingerprint

DATA = ROOT / 'data/synthetic'
REPORT = ROOT / 'report/core'
st.set_page_config(page_title='Cardless Risk Monitor', page_icon='🛡️', layout='wide')
DISPLAY_NAMES = {'score_time': 'Scored at', 'created_at': 'Created at', 'redeemed_at': 'Redeemed at',
                 'voucher_id': 'Voucher', 'customer_id': 'Customer', 'amount': 'Amount (R)', 'risk': 'Risk score',
                 'band': 'Band', 'alert': 'Alert', 'rule_indicators': 'Rule indicators',
                 'scenario_type': 'Simulation scenario', 'is_scam': 'Simulated scam',
                 'event_time': 'Event time', 'event_type': 'Event', 'new_limit': 'New limit (R)', 'channel': 'Channel'}
required = [DATA/'core_scored_vouchers.csv', DATA/'core_dashboard_meta.json', REPORT/'tables/model_comparison.csv']
if not all(p.exists() for p in required):
    st.info('Run notebooks 01, 02 and 03 in order to prepare the dashboard results.')
    st.stop()


@st.cache_data
def load(signatures):
    scores = pd.read_csv(required[0])
    for column in ['created_at', 'redeemed_at', 'score_time']:
        scores[column] = pd.to_datetime(scores[column], format='mixed')
    meta = json.loads(required[1].read_text())
    result = pd.read_csv(required[2])
    if meta['fingerprint'] != fingerprint(ROOT):
        raise ValueError('The saved results no longer match the data or fitting code. Run notebooks 02 and 03 again.')
    return scores, meta, result


@st.cache_data(max_entries=12)
def customer_events(customer_id, cutoff, signature):
    """Read one customer's recent events without showing events after the chosen day."""
    pieces = []
    for chunk in pd.read_csv(DATA/'events_log.csv', chunksize=200000):
        found = chunk[chunk.customer_id == customer_id].copy()
        if not found.empty:
            found['event_time'] = pd.Timestamp(meta['start_date']) + pd.to_timedelta(found.t_min, unit='min')
            found = found[(found.event_time < cutoff) & (found.event_time >= cutoff-pd.Timedelta(days=7))]
            found = found[found.event_type != 'view']
            pieces.append(found)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces).sort_values('event_time', ascending=False)


tracked = required + [DATA/'cardless_features.csv', ROOT/'config/generator.yaml', ROOT/'src/models.py',
                      ROOT/'src/core_evaluation.py', ROOT/'notebooks/02_fit_models.ipynb']
try:
    vouchers, meta, results = load(tuple((str(p), p.stat().st_mtime_ns) for p in tracked))
except ValueError as exc:
    st.error(str(exc)); st.stop()

stage_label = st.sidebar.radio('Scoring point', ['A: voucher creation', 'B: voucher redemption'])
stage = stage_label[0]
part = vouchers[vouchers.stage == stage].copy()
selected = meta['stages'][stage]
st.sidebar.caption(f"Model: {selected['model']}. Equal rule and anomaly weights.")
st.sidebar.caption(f"Alerts start at {selected['cutoffs']['High']:.1f} on the 0–100 risk score. The score is not a fraud probability.")
truth = st.sidebar.toggle('Show simulation ground truth', value=False)
days = sorted(part.score_time.dt.date.unique())
day = st.sidebar.select_slider('Scoring date', options=days, value=days[0])
end = pd.Timestamp(day) + pd.Timedelta(days=1)
today = part[part.score_time.dt.date == day]
st.title('Cardless Risk Monitor')
st.caption('Historical demonstration using saved notebook results. No transactions are blocked by this prototype.')
queue, customer, performance = st.tabs(['Alert queue', 'Customer history', 'Model comparison'])
with queue:
    alerts = today[today.alert]
    cols = st.columns(3)
    cols[0].metric('Vouchers scored', len(today))
    cols[1].metric('Alerts', len(alerts))
    cols[2].metric('Amount under alert', f'R{alerts.amount.sum():,.0f}')
    if truth:
        st.write(f'{int(alerts.is_scam.sum())} of the {int(today.is_scam.sum())} simulated scams were flagged.')
    bands = st.multiselect('Show bands', ['Low', 'Medium', 'High', 'Critical'], default=['High', 'Critical'])
    columns = ['score_time', 'voucher_id', 'customer_id', 'amount', 'risk', 'band', 'alert', 'rule_indicators']
    if truth: columns += ['scenario_type', 'is_scam']
    st.dataframe(today[today.band.isin(bands)].sort_values('risk', ascending=False)[columns].rename(columns=DISPLAY_NAMES), hide_index=True, width='stretch')
    st.caption('Rule indicators show which checklist conditions were present. They are not explanations of the anomaly model. High and Critical correspond to an alert.')
with customer:
    ids = today.sort_values('risk', ascending=False).customer_id.drop_duplicates().tolist()
    if ids:
        cid = st.selectbox('Customer', ids)
        history = part[(part.customer_id == cid) & (part.score_time < end)].sort_values('score_time')
        cols = ['score_time', 'created_at', 'amount', 'risk', 'band', 'alert', 'rule_indicators']
        if stage == 'B': cols.insert(2, 'redeemed_at')
        if truth: cols += ['scenario_type']
        st.dataframe(history[cols].rename(columns=DISPLAY_NAMES), hide_index=True, width='stretch')
        st.caption('Only scored vouchers up to the selected day are shown. Redemption dates are hidden in the creation view.')
        if st.checkbox('Show recent account events (previous seven days)', value=False):
            path = DATA/'events_log.csv'
            if path.exists():
                events = customer_events(cid, end, path.stat().st_mtime_ns)
                if not events.empty:
                    events['event_type'] = events.event_type.str.replace('_', ' ').str.capitalize()
                    st.dataframe(events[['event_time', 'event_type', 'voucher_id', 'amount', 'new_limit', 'channel']].rename(columns=DISPLAY_NAMES), hide_index=True, width='stretch')
                else:
                    st.info('No account events in the preceding seven days.')
    else:
        st.info('No scored vouchers on this date.')
with performance:
    st.caption('Results over the complete test period, not just the selected day. Settings and thresholds were selected on earlier validation records.')
    table = results[results.stage == stage][['model', 'precision', 'recall', 'f1', 'fpr']].copy()
    table['fpr'] *= 100
    table.columns = ['Model', 'Precision', 'Recall', 'F1', 'False positive rate (%)']
    st.dataframe(table.style.format({c:'{:.3f}' for c in table.columns if c != 'Model'}), hide_index=True, width='stretch')
    st.caption('Precision: how many alerts were scams. Recall: how many scams were caught. F1 balances both. False positive rate: how many legitimate vouchers were flagged.')
    for name in ['model_f1', 'scenario_recall']:
        image = REPORT/'figures'/f'{name}.png'
        if image.exists(): st.image(str(image), width='stretch')
