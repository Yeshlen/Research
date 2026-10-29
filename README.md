# Behavioural Risk Scoring for Cardless Cash Withdrawal Fraud

COS700 research project by Yeshlen Moodley.

This project investigates whether customer behaviour can help identify cardless withdrawal scams. It uses synthetic banking activity to compare a Rule Based approach, Isolation Forest, LOF, One-Class SVM and a hybrid modelled using both the Rule Base and One Class SVM approaches.

## Live demo

**[Open the dashboard](https://research-pv84owqnaakhuavatpfysb.streamlit.app/)**

You can view model performance, score existing vouchers, and generate legitimate or scam examples. Vouchers can be scored at creation and redemption. The results describe performance on simulated data.

## Run locally

Using Python 3.14:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

The app automatically downloads the required data and saved models on its first run.

## Research code

- `notebooks/`: data checks and EDA, model fitting, and evaluation. Run notebooks 01–03 in order.
- `src/`: simulation, feature calculation and scoring.
- `config/generator.yaml`: simulation settings.
- `app/`: Streamlit dashboard.
