"""Keep generated voucher features within one visitor's Streamlit session."""
import pandas as pd


def session_examples(state):
    return state.get('generated_examples', pd.DataFrame()).copy()


def remember_example(state, features):
    existing = session_examples(state)
    combined = features.copy() if existing.empty else pd.concat([existing, features], ignore_index=True)
    if combined.voucher_id.duplicated().any():
        raise ValueError('This generated voucher has already been saved.')
    state['generated_examples'] = combined
