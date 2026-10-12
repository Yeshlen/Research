# Dataset Design: Draft for Discussion

Status: draft v1. Based on `cos700_full_chat_summary.md`, the submitted proposal (`inputs/COS700_proposal.pdf`) and the report guide (`inputs/Research_Report_slides.pptx`). Section 10 lists where this design matches or deviates from the proposal.

---

## 1. The short version

- Simulate **customers over time**, not independent rows. Behavioural features such as "amount vs. this customer's average" or "recipient known" are then computed from each customer's real history, so they are consistent by construction.
- Inject **scam episodes** into some customer timelines, plus **legitimate unusual episodes** into others, so that there is genuine overlap.
- Ground truth is a **binary `is_scam`** label plus a **`scenario_type`** column. "Suspicious" becomes an output risk band, not a truth label (see section 5).
- Score at **two points**: voucher creation and voucher redemption. Redemption features do not exist yet at creation time (see section 4). This is the most important design point.
- Scale: about **200k events from about 20k customers over 180 days**, configurable. Fraud prevalence is a parameter, by default about 1.5% scam events.

---

## 2. On "100k+ records"

100k+ is easy to generate, and a larger dataset is fine. Row count alone does not make a dataset a proper ML dataset, though. What matters more:

- **Number of positive cases.** 200k rows at 1.5% gives about 3,000 scam events, which is enough for stable precision/recall estimates and a held-out test set.
- **Realistic prevalence.** Real fraud rates are far below 5%. A 5% fraud rate makes precision look much better than it would in a bank. Reporting results at 2–3 prevalence levels (e.g. 0.5%, 1.5%, 5%) is a cheap and defensible addition.
- **Overlap and realism.** 1 million rows that are trivially separable are worth less than 100k rows that are hard.
- **Per-customer history.** Customer-level features need several events per customer, which is another reason to simulate timelines.

Recommendation: default to 200k events, make it a CLI argument, and state the reasoning in the methodology.

---

## 3. Generation approach: agent-based simulation

No real seed data exists, so SDV, CTGAN and VAE generation have nothing to learn from. A transparent, rule-driven simulation in Python (numpy/pandas, fixed random seed) is the most defensible primary method. It matches section 27.1 of the summary.

### 3.1 Step 1: customer profiles (`customers.csv`)

| Field | How generated (initial idea) |
|---|---|
| `customer_id` | sequential |
| `segment` | e.g. student / salaried / informal-income / pensioner (affects amounts, hours, cardless frequency) |
| `account_age_days` | depends on segment |
| `home_lat`, `home_lon` | sampled around a handful of SA metro centres plus rural spread |
| `cardless_propensity` | how often they use cardless withdrawal (many never, some weekly) |
| `typical_amount` | lognormal by segment |
| `known_recipients` | a small set (1–5) of recurring recipients with their own home areas |
| `active_hours` | preferred login-hour distribution |
| `scam_vulnerability` | latent value, never exported, raises the chance of being targeted |

### 3.2 Step 2: simulate normal behaviour over 180 days

For each customer, sample app sessions and cardless voucher events:
- Mostly sent to known recipients or to self, at typical amounts, during usual hours.
- Redemption happens after a realistic delay (often hours, sometimes next day), near the recipient's area.
- Background events also occur: device changes (new phone), password resets and limit changes, all at low base rates.

### 3.3 Step 3: inject episodes

| scenario_type | is_scam | What it looks like |
|---|---|---|
| `normal` | 0 | everyday behaviour |
| `legit_new_recipient` | 0 | first-time send to someone new, ordinary amount |
| `legit_emergency` | 0 | large, urgent send to family; may follow a limit increase; redeemed fast |
| `legit_new_phone` | 0 | device change then normal use |
| `legit_travel` | 0 | redemption far away because the customer or recipient is travelling |
| `scam_classic` | 1 | call, then login, a new recipient, a high amount, fast creation, redemption within minutes, far away |
| `scam_limit_raise` | 1 | scammer coaches victim to raise limit first, then one or several vouchers |
| `scam_split` | 1 | several smaller vouchers in quick succession to stay under limits |
| `scam_subtle` | 1 | normal-ish amount, redeemed nearby, only 1–2 red flags |

Rule from section 23 of the summary: **no scam scenario triggers every red flag**, and the legit_* scenarios deliberately trigger some of them. Each scenario has a few parameters (probability of each red flag), which go in a config file and the report.

### 3.4 Step 4: derive features from history

All behavioural features are computed from events **before** the current one: rolling averages, counts and whether the recipient was seen before. This removes impossible combinations without needing many manual checks.

---

## 4. Two scoring points (creation vs. redemption)

At voucher creation, `voucher_redeemed`, `redemption_delay_minutes` and `redemption_distance_km` **are not known yet**. A model that uses them while claiming to alert "before cash is withdrawn" is leaking the future. That is the kind of flaw an examiner will spot.

Proposal:

| Stage | When | Features available | Possible action |
|---|---|---|---|
| **A. Creation-time score** | customer submits the voucher | session, history, recipient, amount, recent profile changes | warn, add friction, step-up, delay voucher activation |
| **B. Redemption-time score** | someone tries to redeem at ATM/retailer | everything in A + delay since creation + redemption location | block or hold redemption, call customer |

This maps directly to the intervention points in section 7.2 of the summary. It also gives the results section a better story: "how much can we catch before the voucher exists vs. before the cash leaves?".

---

## 5. Labels: why not normal / suspicious / fraud_like as ground truth

In a simulation the generator knows exactly which events came from a scam episode, so the ground truth is binary. A "suspicious" class has no clear truth meaning. It is either a scam or not, and if the generator decides "suspicious" by thresholding red flags, the models just re-learn the generator's own rule. That is circular.

Recommendation:
- Truth: `is_scam` (0/1) + `scenario_type` (for per-scenario analysis).
- Model output: risk score → bands **Low / Medium (suspicious) / High / Critical**, as in section 30 of the summary.
- "Suspicious" is then reported as "events the model sent to the medium band", which is how a bank would actually use it.

---

## 6. Draft event schema (`cardless_events.csv`)

Identity and time
- `event_id`, `customer_id`, `event_ts`, `hour`, `day_of_week`

Session (known at creation)
- `session_login_ts`, `time_since_login_sec`, `session_screens_visited`, `session_failed_auth_count`
- `device_id_hash`, `is_new_device`, `days_since_device_change`
- `days_since_password_reset`, `days_since_limit_change`, `limit_change_amount`

Transaction (known at creation)
- `amount` (rounded to realistic note multiples), `amount_ratio_to_avg_90d`, `amount_to_daily_limit_ratio`
- `recipient_id_hash`, `recipient_known`, `recipient_first_seen_days`, `is_self_send`
- `prior_cardless_count`, `cardless_count_24h`, `cardless_amount_24h`, `cardless_count_7d`
- `customer_account_age_days`, `customer_segment`

Redemption (known only at stage B)
- `redeemed`, `redemption_delay_min`, `redemption_distance_km`, `redemption_channel` (ATM / retailer)

Truth (never a model input)
- `is_scam`, `scenario_type`, `episode_id`

---

## 7. Validation plan (summary of sections 24–25, adjusted)

- Constraint checks (no redemption fields when not redeemed, no `recipient_known=1` with zero history, limits respected).
- Per-feature distributions by `is_scam` and by `scenario_type`. Checks that overlap exists and that no single feature separates the classes perfectly.
- A quick **"too easy" check**: a shallow decision tree trained on creation-time features. If it gets near 100% recall at near-zero FPR, the scenarios need more overlap.
- Benford first-digit check on amounts, as exploratory only.
- Fixed seed plus config file, so the exact dataset can be regenerated.

---

## 8. Train/test split

Split **by time** (e.g. days 1–120 train, 121–180 test), not randomly. Rolling customer features otherwise leak across the split, and a time split is what a bank would actually do. Unsupervised models train on (mostly) non-scam training events only.

---

## 9. Two output layers: event log + feature table

The proposal (4.2) commits to "realistic mobile banking event **sequences** rather than only isolated transactions", and the prototype (4.1) "simulates events, extracts features, calculates scores and triggers alert decisions". So the generator should output two layers:

1. **`events_log.csv`**: raw, timestamped app events per customer: `login`, `view_balance`, `add_recipient`, `change_limit`, `device_registered`, `password_reset`, `create_voucher`, `redeem_voucher`, `logout`.
2. **`cardless_features.csv`**: one row per voucher, produced by a separate `features.py` that reads only the event log.

Why this is worth it:
- It matches the proposal exactly. The "extracts behavioural features" step becomes real code that can be described in the methodology.
- The dashboard can **replay the event log in time order** and score each voucher as it happens. That is the honest way to back up the "real-time" claim without bank integration.
- Session features such as "went straight to send-cash", "added recipient in same session" and "multiple vouchers in one session" (all named in proposal 4.4) fall out naturally.

---

## 10. Alignment with the submitted proposal

| Proposal says | This design | Note for report |
|---|---|---|
| Event sequences, not isolated transactions (4.2) | Customer timelines + event log (sections 3, 9) | Matches |
| Feature list (4.2) | Superset of it (section 6) | `recent_profile_change` split into device / password / limit |
| `risk_label`: normal / suspicious / fraud_like | `is_scam` truth + `scenario_type`; Low/Medium/High/Critical as **model output** | A refinement. Explain it in Methodology: the 3-tier idea survives as the risk band output. The lecture slides allow "slight deviations". |
| Rules + Isolation Forest (LOF, OCSVM considered) + hybrid (4.3) | Unchanged; the dataset supports all of them | Matches |
| Risk bands 0-39 / 40-69 / 70-89 / 90+ (4.3) | Unchanged | Matches |
| "Specialised LLMs will be used to define, describe and generate the synthetic dataset" (4.4) | LLM-assisted **scenario design and documentation**; rows generated by seeded Python | Word this carefully. LLM-generated rows are not reproducible. Suggested wording: "LLMs supported the design of scenarios and edge cases; the dataset itself was generated by a seeded Python simulator." |
| Interactive dashboard as primary GUI (4.4) | Replay of event log with live scores and alert reasons | This is a **committed deliverable**, not a stretch goal |
| Precision, recall, FPR, detection rate; not accuracy (4.4) | Same + PR-AUC, per-scenario detection | Matches |
| Scenario testing: normal incl. emergency + quick legit redemptions; suspicious incl. multiple vouchers per session (4.4) | `legit_emergency`, `legit_*`, `scam_split` scenarios | Matches |
| Rule: "near or over the daily limit" (4.3) | `amount_to_daily_limit_ratio` | Needs a sourced limit value |

---

## 11. Published product limits (checked 12/10/2026)

| Product | Daily limit | Monthly limit | Other rules | Source |
|---|---|---|---|---|
| Standard Bank Instant Money | R5 000 per day to a receiver; receiver can collect max R5 000/day | R25 000 | Release PIN chosen by sender; 3 wrong PINs → new PIN; 3 years to collect | [Instant Money T&Cs, cl. 4.7, 5.4–5.5, 6.1–6.2](https://sbgmobile.standardbank.co.za/SBGMobileApp/termsandconditions/instantmoney) |
| Absa CashSend | R5 000 (app / online); R500 (cellphone banking) | R25 000 (app); R3 000 (cellphone) | Amounts in R50 increments or rounded to R100; 6-digit access code + 10-digit withdrawal number; valid 30 days | [Absa CashSend page](https://absa.co.za/ways-to-bank/cross-channel-and-messaging-services/cashsend) |
| FNB eWallet | not stated | not stated | Max per send R3 000 (app/online), R1 500 (cellphone/ATM); wallet balance max R5 000; "limits may be reduced" if flagged as potentially fraudulent | [FNB Send Money page](https://www.fnb.co.za/send-money/index.html) |
| Nedbank Cardless Withdrawal | **Default R2 500**, can be raised to max R5 000 in app / online / branch | not stated | Full amount per voucher; valid 30 days; ATM or retailers (Shoprite, Checkers, Pick n Pay, etc.) | [Nedbank Cardless Withdrawal FAQ](https://www.nedbank.co.za/content/dam/nedbank/site-assets/Personal/Digital/Toolsandguidance/FAQs%20-%20Nedbank%20Cardless%20Withdrawal%2020%20October.pdf) |

Proposed simulation parameters (a generic "SA cardless voucher" product, not one bank):
- Default daily limit **R2 500**, customer can raise it to **R5 000** (Nedbank pattern; also makes the `scam_limit_raise` scenario realistic).
- Monthly limit **R25 000** (Standard Bank and Absa agree).
- Amounts in **R50 steps**, often rounded to R100 (Absa rule).
- Voucher valid **30 days**; unredeemed vouchers expire (Absa, Nedbank).
- Full amount redeemed in one go (Nedbank).

The bank pages are undated product pages, so cite them as (Bank, n.d.) with an access date.

---

## 12. Open questions for Yeshlen

1. Agree on binary truth + scenario type instead of the three-class label?
2. Agree on two scoring points (creation and redemption)?
3. Size and prevalence: 200k events / ~1.5% scam as default, with a prevalence sensitivity run?
4. ~~Product limits~~: researched, see section 11.
5. Does the proposal commit to anything specific about the data (size, features, labels) that this would contradict?
