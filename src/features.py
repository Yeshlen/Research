"""
Streaming feature extraction from the raw event log.

Events are replayed in time order per customer. A running profile per customer is updated after every
event, exactly as a bank's real-time scoring service would keep one. Features for a voucher are taken
from the profile *at the moment the event happens*, so no feature can use information from the future.

Two scoring points:
    Stage A (creation):   features known when the customer submits the voucher
    Stage B (redemption): stage A + features known when someone tries to cash the voucher out

Usage:
    python src/features.py --data data/synthetic
writes data/synthetic/cardless_features.csv
"""
import argparse
import math
import time
from collections import deque

import numpy as np
import pandas as pd
import yaml

MIN_PER_DAY = 1440
NO_HISTORY_DAYS = 365          # sentinel for "never happened" in days-since features


def km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


class Profile:
    """Everything the bank knows about one customer up to 'now'."""

    def __init__(self, home_lat, home_lon, account_open_min, limit):
        self.home = (home_lat, home_lon)
        self.account_open_min = account_open_min
        self.limit = float(limit)              # current daily cardless limit
        self.vouchers = deque()                # (t, amount) for the last 90 days
        self.n_vouchers = 0
        self.max_amount = 0.0
        self.last_voucher_t = None
        self.recipients = set()
        self.hour_sin = self.hour_cos = 0.0
        self.redeem_lat_sum = self.redeem_lon_sum = 0.0
        self.n_redeemed = 0
        self.last_limit_change = None          # (t, increase)
        self.last_device_change = None
        self.last_password_reset = None
        # current session
        self.session = None
        self.login_t = None
        self.screens = 0
        self.session_vouchers = 0
        self.session_added = set()

    def prune(self, t):
        while self.vouchers and self.vouchers[0][0] < t - 90 * MIN_PER_DAY:
            self.vouchers.popleft()


def since_days(t, prev):
    return NO_HISTORY_DAYS if prev is None else min((t - prev) / MIN_PER_DAY, NO_HISTORY_DAYS)


def creation_features(p, e, start):
    t, amount = e.t_min, float(e.amount)
    p.prune(t)
    amounts = [a for _, a in p.vouchers]
    avg = np.mean(amounts) if amounts else None
    limit = p.limit
    hour = (t % MIN_PER_DAY) / 60
    if p.n_vouchers:
        mean_angle = math.atan2(p.hour_sin, p.hour_cos)
        diff = abs(hour / 24 * 2 * math.pi - mean_angle) % (2 * math.pi)
        hour_dev = min(diff, 2 * math.pi - diff) / (2 * math.pi) * 24
    else:
        hour_dev = 0.0
    ts = start + pd.Timedelta(minutes=t)
    lc = p.last_limit_change
    return {
        "voucher_id": int(e.voucher_id),
        "customer_id": e.customer_id,
        "created_at": ts.floor("s"),
        "day": int(t // MIN_PER_DAY),
        # amount
        "amount": amount,
        "amount_ratio_to_avg_90d": round(amount / avg, 3) if avg else 1.0,
        "amount_ratio_to_max": round(amount / p.max_amount, 3) if p.max_amount else 1.0,
        "amount_to_limit_ratio": round(amount / limit, 3),
        # history
        "history_count": p.n_vouchers,
        "has_history": int(p.n_vouchers > 0),
        "days_since_last_voucher": round(since_days(t, p.last_voucher_t), 2),
        "vouchers_24h": sum(1 for tt, _ in p.vouchers if tt >= t - MIN_PER_DAY),
        "amount_24h": sum(a for tt, a in p.vouchers if tt >= t - MIN_PER_DAY),
        "vouchers_7d": sum(1 for tt, _ in p.vouchers if tt >= t - 7 * MIN_PER_DAY),
        "vouchers_in_session": p.session_vouchers,
        # recipient
        "recipient_known": int(e.recipient_id in p.recipients),
        "is_self_send": int(e.recipient_id == "SELF"),
        "recipient_added_in_session": int(e.recipient_id in p.session_added),
        # session behaviour
        "secs_since_login": round((t - p.login_t) * 60, 1) if p.login_t is not None else -1,
        "screens_before_create": p.screens,
        # timing
        "hour": round(hour, 2),
        "hour_deviation": round(hour_dev, 2),
        "is_night": int(hour < 6 or hour >= 21),
        "day_of_week": ts.dayofweek,
        "is_payday_window": int(ts.day >= 25 or ts.day == 1),
        # account changes
        "limit_changed_24h": int(lc is not None and t - lc[0] <= MIN_PER_DAY),
        "limit_increase_24h": lc[1] if lc is not None and t - lc[0] <= MIN_PER_DAY else 0,
        "days_since_device_change": round(since_days(t, p.last_device_change), 2),
        "new_device_7d": int(p.last_device_change is not None and t - p.last_device_change <= 7 * MIN_PER_DAY),
        "password_reset_7d": int(p.last_password_reset is not None and t - p.last_password_reset <= 7 * MIN_PER_DAY),
        "account_age_days": round((t - p.account_open_min) / MIN_PER_DAY, 1),
    }


def redemption_features(p, e, created_t):
    lat, lon = float(e.lat), float(e.lon)
    if p.n_redeemed:
        usual = (p.redeem_lat_sum / p.n_redeemed, p.redeem_lon_sum / p.n_redeemed)
    else:
        usual = p.home
    hour = (e.t_min % MIN_PER_DAY) / 60
    return {
        "redeemed": 1,
        "redeem_delay_min": round(e.t_min - created_t, 2),
        "redeem_dist_home_km": round(km(lat, lon, *p.home), 1),
        "redeem_dist_usual_km": round(km(lat, lon, *usual), 1),
        "redeem_hour": round(hour, 2),
        "redeem_is_night": int(hour < 6 or hour >= 21),
        "redeem_at_atm": int(e.channel == "ATM"),
    }


def extract(data_dir, cfg):
    start = pd.Timestamp(cfg["start_date"])
    events = pd.read_csv(f"{data_dir}/events_log.csv", dtype={"recipient_id": str, "channel": str})
    customers = pd.read_csv(f"{data_dir}/customers.csv", parse_dates=["account_open_date"]).set_index("customer_id")
    events = events.sort_values(["customer_id", "t_min"], kind="stable")

    rows = {}
    created_t = {}
    profiles = {}
    for e in events.itertuples(index=False):
        p = profiles.get(e.customer_id)
        if p is None:
            c = customers.loc[e.customer_id]
            open_min = (c.account_open_date - start).total_seconds() / 60
            p = profiles[e.customer_id] = Profile(c.home_lat, c.home_lon, open_min, c.daily_limit_at_start)
        et, t = e.event_type, e.t_min
        if et == "login":
            p.session, p.login_t, p.screens, p.session_vouchers, p.session_added = e.session_id, t, 0, 0, set()
        elif et == "view":
            p.screens += 1
        elif et == "add_recipient":
            p.session_added.add(e.recipient_id)
        elif et == "change_limit":
            old = p.limit
            p.limit = float(e.new_limit)
            p.last_limit_change = (t, max(0.0, p.limit - old))
        elif et == "device_registered":
            p.last_device_change = t
        elif et == "password_reset":
            p.last_password_reset = t
        elif et == "create_voucher":
            rows[int(e.voucher_id)] = creation_features(p, e, start)
            created_t[int(e.voucher_id)] = t
            amount, hour = float(e.amount), (t % MIN_PER_DAY) / 60
            p.vouchers.append((t, amount))
            p.n_vouchers += 1
            p.max_amount = max(p.max_amount, amount)
            p.last_voucher_t = t
            p.recipients.add(e.recipient_id)
            p.session_vouchers += 1
            p.hour_sin += math.sin(hour / 24 * 2 * math.pi)
            p.hour_cos += math.cos(hour / 24 * 2 * math.pi)
        elif et == "redeem_voucher":
            vid = int(e.voucher_id)
            rows[vid].update(redemption_features(p, e, created_t[vid]))
            p.redeem_lat_sum += float(e.lat)
            p.redeem_lon_sum += float(e.lon)
            p.n_redeemed += 1

    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(drop=True)
    df["redeemed"] = df["redeemed"].fillna(0).astype(int)
    truth = pd.read_csv(f"{data_dir}/vouchers_truth.csv")
    df = df.merge(truth[["voucher_id", "is_scam", "scenario_type", "episode_id"]], on="voucher_id", how="left")
    df["split"] = np.where(df["day"] < cfg["train_days"], "train", "test")
    return df.sort_values("created_at").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/generator.yaml")
    ap.add_argument("--data", default="data/synthetic")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    t0 = time.time()
    df = extract(a.data, cfg)
    out = f"{a.data}/cardless_features.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} vouchers, {df.is_scam.sum()} scam ({df.is_scam.mean():.2%}) -> {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
