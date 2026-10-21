# Revised EDA findings

Source: executed `notebooks/01_data_and_eda.ipynb`, seed 42. Tables and plots are in `report/core/`. The former extended EDA is archived.

- 216,147 vouchers; 19,078 active customers; 2,761 scam vouchers (1.28%); 209,222 redemptions.
- All 20 product, history, duplication and truth-label checks returned zero violations.
- Class comparisons use activity before day 120, rather than later test records.
- Median creation amounts: legitimate R600; scam R1,800. Median ratios to the 90-day average: 1.00 and 1.684.
- Under-15-minute redemptions: 9.14% legitimate; 58.54% scam, among earlier redemptions.
- New recipients: 44.97% legitimate; 68.61% scam. Recent limit changes: 1.32% and 35.15%. Recent devices: 1.65% and 5.75%.

These are observations of the configured simulator. Differences and overlap help explain the comparison; passing checks does not establish real-world realism. The notebook displays three plots: amounts, redemption delays and warning-indicator rates.
