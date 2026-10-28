# Public demonstration deployment

The repository supplies the application, fixed-version dependencies and saved model bundle. Raw data is attached to the `demo-data-v1` GitHub Release as `cardless-dataset-seed42-v1.tar.gz`.

## Deploy

1. Make the GitHub repository public. The release download URL must work while signed out.
2. Sign in to Streamlit Community Cloud and choose Create app.
3. Select `Yeshlen/Research`, branch `main`, and `app/dashboard.py`.
4. In Advanced settings select Python **3.14**. The model bundle was saved with Python 3.14.3 and the pinned NumPy, SciPy, scikit-learn and joblib versions.
5. Deploy, then check every model page, both scoring points, record generation and the CSV download.
6. Copy the live app URL to the README and the repository's About → Website field.

The app downloads the release archive only when required data files are missing. It verifies the archive checksum and every extracted file, rejects unexpected paths, and does not overwrite existing research data. The archive contains voucher features, customers, customer habits, event history, generation metadata and voucher truth labels. It excludes old score exports and temporary demonstrations.

The full dataset is used for browsing and independent generation. Fitting never runs at app startup. `dataset_metadata.json` retains the configuration that generated the original records. Restoring the current YAML seed to 42 does not rewrite the already generated data or the historical generation settings.

## Visitor data

Each visitor's generated features are held in their Streamlit session. They can revisit these vouchers on other model pages or download a CSV. No shared server CSV is written. Session data can disappear when a session ends or the service restarts. Do not promise permanent storage.

## Update or remove

Pushing changes to `main` updates the deployment. When fitting inputs change, rebuild models and the associated dataset release together; do not reuse a mismatched bundle. Keep old release assets unchanged and publish a new version with a new manifest for changed data.

To rebuild the same archive locally:

```bash
python scripts/package_demo_data.py --output /tmp/cardless-dataset-seed42-v1.tar.gz
```

This packages existing files; it does not regenerate data. Upload a new archive only after checking that it belongs with the models being deployed.

The public app can be deleted from Community Cloud after the demonstration period without deleting the repository. Free hosting has limited resources and sleeps after inactivity, so test the public link before displaying it on printed material.
