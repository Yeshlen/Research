# COS700: Behavioural Risk Scoring for Cardless Cash Withdrawal Fraud

University of Pretoria COS700 research project (supervisor: Prof Jan Eloff).

**[Try the live demo](https://research-pv84owqnaakhuavatpfysb.streamlit.app/)** — explore the models, score vouchers and generate new examples.

A prototype that simulates mobile banking events, extracts behavioural features and scores cardless withdrawal vouchers for possible social-engineering scams.

## Setup and notebook workflow

Create a Python environment and install `requirements.txt`. Select that environment as the notebook kernel, then run all cells in each notebook in order:

| Notebook | What it does |
|---|---|
| `notebooks/01_data_and_eda.ipynb` | Prepare/reuse data, check records, describe scenarios and draw three EDA plots |
| `notebooks/02_fit_models.ipynb` | Create and fit models; choose settings, alert thresholds and one hybrid using validation records |
| `notebooks/03_results_and_dashboard.ipynb` | Evaluate fixed choices, draw results plots and export dashboard scores |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Notebook 01 reuses an existing dataset. Set its `REGENERATE` flag only when rebuilding the simulator data. Synthetic data files are excluded from Git. The saved fitted-model bundle is supplied through the GitHub Release for deployment; run the preceding notebooks when rebuilding data or models. Outputs are saved under `report/core/`, with fitted models in `report/core/models/`.

## Dataset simulator

The active entry point is `src/simulate.py`. Its generation stages are in `src/simulator/`: customer profiles, ordinary activity, special scenarios, event recording and CSV output. Start with `src/simulator/run.py` to follow the process. Historical implementations are retained locally and are not needed to run this repository.

For a small example in a separate folder:

```bash
python src/simulate.py --customers 10 --out data/simulator_demo
python src/features.py --data data/simulator_demo
```

Notebook 01 uses the same `simulator.generate_dataset` function when regeneration is enabled.

## Revised comparison

Exactly five approaches at voucher creation and redemption: rules, Isolation Forest, LOF, One-Class SVM, and one equal-weight hybrid combining rules with the best validation anomaly model. Fit 90 days, validate 30 days, test 60 days. Creation uses creation dates; redemption uses redemption dates. Model settings and alert thresholds are selected by validation F1 before the test period is scored.

Results report precision, recall, F1, false positive rate, underlying decision counts and scenario rates. The anomaly models fit without labels, but validation selection uses scam labels. The data is synthetic and was already examined during development.

Simulation settings are in `config/generator.yaml`. Feature calculations are in `src/features.py`; model inputs and rules are in `src/models.py`.

## Dashboard and checks

After running all three notebooks:

```bash
streamlit run app/dashboard.py
pytest -q
```

The dashboard has pages for Rule Based, Isolation Forest, LOF, One-Class SVM and Hybrid, plus model comparison and independent record generation. Each model can score an existing test voucher or a generated example from their session. The outline dice button chooses a random test record from the current filter. Generate Record creates legitimate/scam activity (or a 50/50 random choice), scores it with fixed models and opens a review-alert modal when the saved threshold is crossed. Generated examples stay within each visitor's session and can be downloaded as CSV; they never change the research history or saved test metrics. Creation and redemption remain separate scoring points. Rule indicators provide context for anomaly models, not internal model explanations.

The notebooks are the main analysis workflow. Reusable Python modules remain in `src/` for simulation, feature extraction, scoring and checks; the application remains in `app/`.

## Streamlit Community Cloud

Use repository `Yeshlen/Research`, branch `main`, entry point `app/dashboard.py`, and Python **3.14** in Advanced settings. `requirements.txt` pins the versions used by the saved models. No API keys or paid services are needed for the public demo.

When the required files are absent, the app downloads the original synthetic dataset, saved models and model-comparison table from the `demo-data-v1` GitHub Release. `deployment/data_release.json` and `deployment/runtime_release.json` record download URLs and checksums. The `report/` folder is local and ignored by Git; the hosted app restores only its required model bundle and results table. The repository and release must be public for anonymous Community Cloud downloads. Existing local data is reused without downloading or overwriting it. The bundled dataset and existing reported metrics are unchanged.

Generated examples stay in each visitor's session. Download the CSV before leaving to retain them. The app does not save visitors' generations to shared server CSVs. Community Cloud may sleep after inactivity, and the first visit may take longer while the app starts or downloads data.

