"""
Illustrative sketch of the cardless-withdrawal simulator.
NOT the final generator: one scam type, no event log, few features.
It shows the mechanics: customer -> timeline -> vouchers -> scam episode -> features.

Run:  python generator_sketch.py --customers 500 --seed 42
"""
import argparse
import numpy as np
import pandas as pd

# ---- Config: every number here becomes a row in the report's parameter table ----
CFG = {
    "days": 180,
    "daily_limit_default": 2500,      # Nedbank default (Nedbank, n.d.)
    "daily_limit_max": 5000,          # SB / Absa / Nedbank max
    "amount_step": 50,                # Absa R50 increments
    "segments": {                     # share, median vouchers/month, median amount
        "student":    (0.20, 0.5, 300),
        "salaried":   (0.45, 0.8, 800),
        "informal":   (0.20, 1.2, 600),
        "pensioner":  (0.15, 0.4, 500),
    },
    "rate_sigma": 1.0,                # log-normal spread of activity -> long tail
    "amount_sigma": 0.5,              # spread of a customer's typical amount between customers
    "within_amount_sigma": 0.35,      # spread of one customer's amounts around their typical
    "payday_boost": 2.0,              # 25th to 1st
    "p_targeted": 0.02,               # chance a customer is hit by a scam in the period
    "scam_flag_probs": {              # each red flag fires independently: not all at once
        "new_recipient": 0.85, "limit_raise": 0.5, "fast_redeem": 0.8,
        "far_redeem": 0.6, "quick_after_login": 0.7, "high_amount": 0.75,
    },
}
METROS = {  # (lat, lon, population weight)
    "JHB": (-26.20, 28.05, 0.30), "CPT": (-33.92, 18.42, 0.20),
    "DBN": (-29.86, 31.02, 0.17), "PTA": (-25.75, 28.19, 0.15),
    "PE":  (-33.96, 25.60, 0.08), "BFN": (-29.12, 26.21, 0.10),
}


def km(a, b):
    """Rough distance in km between two (lat, lon) points."""
    dlat = (a[0] - b[0]) * 111
    dlon = (a[1] - b[1]) * 111 * np.cos(np.radians(a[0]))
    return float(np.hypot(dlat, dlon))


def round_amount(x, limit):
    return float(np.clip(round(x / CFG["amount_step"]) * CFG["amount_step"], CFG["amount_step"], limit))


def make_customer(cid, rng):
    seg_names = list(CFG["segments"])
    seg = rng.choice(seg_names, p=[CFG["segments"][s][0] for s in seg_names])
    _, med_rate, med_amt = CFG["segments"][seg]
    metro = rng.choice(list(METROS), p=[m[2] for m in METROS.values()])
    home = (METROS[metro][0] + rng.normal(0, 0.15), METROS[metro][1] + rng.normal(0, 0.15))
    n_recip = 1 + rng.poisson(1.5)
    recipients = [(f"R{cid}_{k}", (home[0] + rng.normal(0, 0.3), home[1] + rng.normal(0, 0.3)))
                  for k in range(n_recip)]
    return {
        "customer_id": cid, "segment": seg, "home": home,
        "monthly_rate": min(med_rate * rng.lognormal(0, CFG["rate_sigma"]), 20),  # long tail, capped
        "typical_amount": med_amt * rng.lognormal(0, CFG["amount_sigma"]),
        "pref_hour": float(np.clip(rng.normal(13, 3), 6, 22)),
        "p_known": rng.beta(8, 2),                                         # mostly sends to known people
        "recipients": recipients,
        "start_day": int(rng.integers(0, 60)) if rng.random() < 0.15 else 0,  # some join late
    }


def simulate_normal(c, rng):
    """Poisson process day by day with payday boost."""
    rows = []
    for day in range(c["start_day"], CFG["days"]):
        dom = (day % 30) + 1
        rate = c["monthly_rate"] / 30 * (CFG["payday_boost"] if dom >= 25 or dom == 1 else 1)
        for _ in range(rng.poisson(rate)):
            if rng.random() < c["p_known"]:
                rid, rloc = c["recipients"][rng.integers(len(c["recipients"]))]
            else:
                rid, rloc = f"NEW_{rng.integers(1e9)}", (c["home"][0] + rng.normal(0, 0.5), c["home"][1] + rng.normal(0, 0.5))
            hour = float(np.clip(rng.normal(c["pref_hour"], 2), 0, 23.9))
            redeem_loc = (rloc[0] + rng.normal(0, 0.02), rloc[1] + rng.normal(0, 0.02))
            rows.append({
                "customer_id": c["customer_id"], "ts": day * 24 + hour,
                "amount": round_amount(c["typical_amount"] * rng.lognormal(0, CFG["within_amount_sigma"]), CFG["daily_limit_default"]),
                "recipient": rid,
                "mins_since_login": rng.lognormal(np.log(4), 0.6),
                "redeem_delay_min": rng.lognormal(np.log(180), 1.0),      # median 3 hours, wide spread
                "redeem_km_from_home": km(c["home"], redeem_loc),
                "limit_raised_24h": 0, "is_scam": 0, "scenario": "normal",
            })
    return rows


def inject_scam(c, rng):
    """One scam episode: each red flag fires with its own probability."""
    f = {k: rng.random() < p for k, p in CFG["scam_flag_probs"].items()}
    day = int(rng.integers(max(c["start_day"], 30), CFG["days"]))
    limit = CFG["daily_limit_max"] if f["limit_raise"] else CFG["daily_limit_default"]
    amount = rng.uniform(0.7, 1.0) * limit if f["high_amount"] else c["typical_amount"] * rng.lognormal(0, 0.3)
    far = (c["home"][0] + rng.normal(0, 3), c["home"][1] + rng.normal(0, 3)) if f["far_redeem"] else (c["home"][0] + rng.normal(0, 0.1), c["home"][1] + rng.normal(0, 0.1))
    rid = f"NEW_{rng.integers(1e9)}" if f["new_recipient"] else c["recipients"][0][0]
    return [{
        "customer_id": c["customer_id"], "ts": day * 24 + float(rng.uniform(8, 18)),
        "amount": round_amount(amount, limit), "recipient": rid,
        "mins_since_login": rng.uniform(0.5, 2.5) if f["quick_after_login"] else rng.lognormal(np.log(4), 0.6),
        "redeem_delay_min": rng.uniform(2, 15) if f["fast_redeem"] else rng.lognormal(np.log(90), 0.8),
        "redeem_km_from_home": km(c["home"], far),
        "limit_raised_24h": int(f["limit_raise"]), "is_scam": 1, "scenario": "scam_classic",
    }]


def add_history_features(df):
    """Customer-baseline features computed from PAST events only (no leakage)."""
    df = df.sort_values(["customer_id", "ts"]).reset_index(drop=True)
    g = df.groupby("customer_id")
    df["history_count"] = g.cumcount()
    df["avg_amount_before"] = g["amount"].transform(lambda s: s.shift().expanding().mean())
    df["amount_ratio_to_avg"] = (df["amount"] / df["avg_amount_before"]).round(2)
    seen = []
    for _, grp in g:
        known = set()
        for r in grp["recipient"]:
            seen.append(int(r in known))
            known.add(r)
    df["recipient_known"] = seen
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sketch_output.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = []
    for cid in range(a.customers):
        c = make_customer(cid, rng)
        rows += simulate_normal(c, rng)
        if rng.random() < CFG["p_targeted"]:
            rows += inject_scam(c, rng)
    df = add_history_features(pd.DataFrame(rows))
    df["redeem_delay_min"] = df["redeem_delay_min"].round(1)
    df["mins_since_login"] = df["mins_since_login"].round(1)
    df["redeem_km_from_home"] = df["redeem_km_from_home"].round(1)
    df.to_csv(a.out, index=False)

    per_cust = df.groupby("customer_id").size()
    print(f"{len(df)} vouchers from {df.customer_id.nunique()} customers, {df.is_scam.sum()} scam")
    print("vouchers per customer: ", per_cust.describe(percentiles=[.25, .5, .75, .95]).round(1).to_dict())


if __name__ == "__main__":
    main()
