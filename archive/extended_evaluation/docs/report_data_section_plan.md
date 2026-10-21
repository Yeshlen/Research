# Report Plan: Data Generation and Preparation Section

Target: 4–5 pages. In the required structure this sits inside **Methodology** (how the data was built) with the EDA graphs either here or at the start of **Discussion/Results**. The lecture slides allow "slight deviations", so a dedicated subsection is fine.

Writing rule (from the earlier chats): every reference must do a job in the text. Each decision below names the paper that defends it.

---

## Outline

| # | Subsection | ~Pages | Content |
|---|---|---|---|
| 1 | Why synthetic data | 0.5 | Real fraud data is confidential (POPIA); no public dataset matches vishing-driven cardless withdrawals; PaySim/BankSim/BAF reviewed and why they were not used directly |
| 2 | Simulation design | 1 | Agent-based simulation of customers over 180 days; customer segments; product rules taken from published bank limits; scam and legitimate-unusual episodes |
| 3 | Feature engineering | 1–1.5 | Table of every feature: definition, stage (creation / redemption), why it matters, source backing it. Customer-baseline ("relative to own normal") features explained |
| 4 | Labels and split | 0.5 | Binary truth + scenario type; time-based train/test split; why no random split |
| 5 | Validation and EDA | 1–1.5 | Constraint checks, distributions by label, overlap, "too easy" check, Benford's Law (exploratory), correlation heatmap |

---

## Decision → reference map

| Decision | What the paper supports | Reference |
|---|---|---|
| Use simulation-based synthetic data for fraud research | Agent-based simulator of mobile-money transactions built because real financial data is private; the source of the PaySim dataset | Lopez-Rojas, Elmir & Axelsson (2016). *PaySim: A financial mobile money simulator for fraud detection.* EMSS 2016. |
| Same, for bank payments | Earlier simulator for bank payment fraud research | Lopez-Rojas & Axelsson (2014). *BankSim: A bank payments simulator for fraud detection research.* EMSS 2014. |
| Simulate customers with profiles, generate transactions over time, inject fraud scenarios | Their open handbook builds a transaction simulator with customer spending profiles, locations and injected fraud scenarios, then derives windowed features from it. Very close to this design | Le Borgne, Siblini, Lebichot & Bontempi (2022). *Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook.* Université Libre de Bruxelles. |
| Privacy-preserving synthetic benchmark as accepted practice; imbalanced, time-dependent data | NeurIPS Bank Account Fraud dataset suite | Jesus et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.* NeurIPS Datasets & Benchmarks. |
| Customer-baseline features (amount vs. own average, frequency windows) | Shows aggregated/periodic features per customer substantially improve fraud detection | Correa Bahnsen, Aouada, Stojanovic & Ottersten (2016). *Feature engineering strategies for credit card fraud detection.* Expert Systems with Applications, 51, 134–142. |
| Same (aggregation over history) | Transaction aggregation as a fraud detection strategy | Whitrow, Hand, Juszczak, Weston & Adams (2009). *Transaction aggregation as a strategy for credit card fraud detection.* Data Mining and Knowledge Discovery, 18(1), 30–55. |
| Behavioural profiling / "compare to the customer's normal" | Already in proposal | Bolton & Hand (2002); Chandola, Banerjee & Kumar (2009) |
| Time-based split, imbalance, practical evaluation | Lessons from real fraud systems: concept drift, delayed labels, class imbalance | Dal Pozzolo, Caelen, Le Borgne, Waterschoot & Bontempi (2014). *Learned lessons in credit card fraud detection from a practitioner perspective.* Expert Systems with Applications, 41(10), 4915–4928. |
| Only past events in features (no future leakage) | Defines and classifies data leakage | Kaufman, Rosset, Perlich & Stitelman (2012). *Leakage in data mining: Formulation, detection, and avoidance.* ACM TKDD, 6(4). |
| Combining rules / unsupervised scores with other signals (hybrid) | Unsupervised outlier scores added as features improve fraud detection | Carcillo, Le Borgne, Caelen, Kessaci, Oblé & Bontempi (2021). *Combining unsupervised and supervised learning in credit card fraud detection.* Information Sciences, 557, 317–331. (This was cut from the proposal for being unused; here it has a real job.) |
| PR-AUC rather than accuracy / ROC on rare classes | PR curves are more informative than ROC on imbalanced data | Saito & Rehmsmeier (2015). *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.* PLoS ONE, 10(3). |
| Why not CTGAN / SDV as the generator | These learn from a real seed dataset, which does not exist here; mention as future work | Patki, Wedge & Veeramachaneni (2016). *The Synthetic Data Vault.* IEEE DSAA. Xu, Skoularidou, Cuesta-Infante & Veeramachaneni (2019). *Modeling Tabular Data using Conditional GAN.* NeurIPS. |
| Benford's Law as an exploratory check | Standard reference for Benford in auditing/fraud | Nigrini (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection.* Wiley. |
| Product limits, voucher expiry, R50 steps | Published bank product rules | Standard Bank (n.d.) Instant Money T&Cs; Absa (n.d.) CashSend; Nedbank (n.d.) Cardless Withdrawal FAQ; FNB (n.d.) Send Money. See `dataset_design_draft.md` section 11. |
| Scam modus operandi (limit raise, read back PIN) | Bank scam warnings, SABRIC | Standard Bank (n.d.-b); SABRIC (2025), already in proposal |

All of these are real, published works. I confirmed PaySim, the Le Borgne handbook, Correa Bahnsen (2016) and Carcillo (2021) by search. Add each to Zotero/JabRef from its DOI or publisher page, rather than copying the details above, so the BibTeX is exact.

---

## Figures planned for this section

1. Simulation pipeline diagram (customers → timelines → episodes → event log → features)
2. Scam episode timeline example (login → limit raise → voucher → fast redemption)
3. Label / scenario distribution
4. Amount distribution by `is_scam` (log scale) + `amount_ratio_to_avg` by label
5. Redemption delay by label
6. Time since login by label
7. Correlation heatmap
8. Benford first-digit chart vs. expected
