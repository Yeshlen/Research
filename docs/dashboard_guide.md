# Dashboard guide

From the Research folder, activate the project environment and start the app:

```bash
source .venv/bin/activate
streamlit run app/dashboard.py
```

The existing dataset and saved models must be present. Run notebooks 01, 02 and 03 in order if setting up a new checkout. Ordinary dashboard use does not fit or tune models.

## Pages

- The home page introduces the five methods, including Rule Based.
- Each model page explains the method and lets the user score a voucher. Choose **Existing test records** or **Generated examples**. The Performance tab shows saved test precision, recall, F1, false positive rate and a confusion matrix.
- **Generate Record** generates independent examples and scores them immediately at creation. **Simulate redemption** applies the redemption-stage model to the eligible vouchers in that example.
- **Model Comparison** shows the fixed test-period metrics. Generated demonstrations are excluded.

## Generation controls

- Select **Legitimate**, choose a scenario, then press **Generate** to create ordinary legitimate activity by default. Emergency, new-phone and travel scenarios can also be selected.
- Select **Scam**, choose a scenario, then press **Generate** to create a scam episode. Automatic uses the original configured scam-scenario weights; an individual scenario can be selected.
- Select **Random (50/50)** and press **Generate** for an equal-probability choice between an ordinary legitimate session and a scam from the original configured scenario mix. This is a demonstration mix, not the prevalence of the research dataset.

The scenario decides the simulation label. It does not determine whether the model raises an alert. There is no retry based on the model's score. A customer with no remaining allowance may create no voucher; the app reports this rather than bypassing the product limits.

A split-voucher episode may create two to four vouchers, subject to remaining allowance. Scenario controls appear only for the selected activity type. No scenario controls are shown before selection or for Random.

An ordinary session can also create a second voucher using the existing simulation logic. Later vouchers within the example include its earlier vouchers in their features.

## Existing history and independence

For each call, the app chooses an existing customer and a history checkpoint in the test portion of the original window. Only original events strictly before that checkpoint are copied. The original planners generate an independent continuation, which can occur later in the observation window. Original events after the checkpoint do not enter this branch.

The saved generation configuration is used when available. Customer details and stored latent habits come from `customers.csv` and `customers_latent.csv`. Current limits, earlier spending and recipient use are restored from preceding events. The original hidden regular-recipient lists were not persisted, so the demo reconstructs a list from observed earlier recipients; if none exists it creates a fictional recipient that remains unknown in the observed history. Stored habits are rounded CSV values. This is new demonstration sampling using the same planners, not exact replay of the original generator's hidden state.

The shared event recorder accepts optional starting state for this purpose. Its normal dataset-generation path is unchanged and is covered by the original byte-equivalence tests. The existing feature extractor runs on a temporary copy of the customer and events. No full dataset regeneration is needed.

Each new example starts from the original files again. Earlier generated examples never become history for later generations. Within one episode, the new vouchers update its temporary profile in event order.

## Generated examples

The dashboard stores generated voucher features in the current visitor's session, separately from other visitors. They remain available across model pages during that session and can be downloaded as CSV. Refreshing or ending a session can clear them. Original research files, fitted models and test tables are read only.

The reusable `save_example` function in `src/demo_simulation.py` is still available for local scripts and tests; the public dashboard does not call it or read shared `data/demo/` files.

Redemption information is planned by the generator for later demonstration. Creation scoring never uses it. Selecting redemption scoring or pressing **Simulate redemption** reveals and scores that stage. Some vouchers have no redemption within the observation window and can only be scored at creation.

## Alerts

The selected method uses its saved validation threshold. An alert opens a modal and remains visible on the page after dismissal. The modal lists the score, threshold and triggered checklist conditions. It is a request for review, not confirmation of fraud, and no payment is blocked.

For Rule Based, the checklist explains the points. For LOF, One-Class SVM and Isolation Forest, it supplies context only; it is not a feature attribution for the model. The Hybrid display also shows its rule score and anomaly percentile. Scores are not scam probabilities.

The simulated truth is shown separately. This lets users see correct alerts, missed scams and false alerts. Closing the modal does not discard the result, and ordinary page reruns do not repeatedly open it.

## Code layout

| File | Role |
|---|---|
| `app/dashboard.py` | Home page and existing launch entry point |
| `app/pages/` | Simple model, simulation and comparison pages |
| `app/session_examples.py` | Visitor-specific generated voucher storage |
| `app/utils.py` | Shared loading, voucher selection, results and modal |
| `src/demo_simulation.py` | Independent customer continuation and separate CSV storage |
| `src/demo_scoring.py` | Saved-model scoring, without labels or fitting |

The original saved-model fingerprint checks remain active. A mismatch means the fitting data or relevant settings/code changed; rerun notebooks 02 and 03 rather than silently using stale models.
