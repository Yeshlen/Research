"""
Agent-based simulator for cardless cash withdrawal (CashSend / eWallet / Instant Money style) activity.

Each simulated customer gets their own habits, drawn from distributions in config/generator.yaml.
The simulator then plays out their mobile banking over the observation window, injects legitimate
"unusual" episodes and social-engineering scam episodes, and writes a raw, timestamped app event log.

Outputs (in --out directory):
    events_log.csv        one row per app event (login, view, add_recipient, change_limit,
                          device_registered, password_reset, create_voucher, redeem_voucher, logout)
    vouchers_truth.csv    ground truth per voucher (is_scam, scenario_type, episode_id); never a model input
    customers.csv         customer attributes a bank would know (home, account open date)
    customers_latent.csv  hidden simulation parameters, kept only for analysis of the generator

Usage:
    python src/simulate.py --config config/generator.yaml --out data/synthetic
"""
import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
import yaml

MIN_PER_DAY = 1440
EVENT_COLS = ["customer_id", "session_id", "t_min", "event_type", "device_id", "voucher_id",
              "recipient_id", "amount", "new_limit", "lat", "lon", "channel"]
SELF = "SELF"


class Ids:
    """Global id counters shared across customers."""

    def __init__(self):
        self.session = 0
        self.voucher = 0
        self.recipient = 0
        self.device = 0
        self.episode = 0

    def next(self, name):
        setattr(self, name, getattr(self, name) + 1)
        return getattr(self, name)


class Simulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])
        self.ids = Ids()
        self.metro_names = list(cfg["metros"])
        m = np.array([cfg["metros"][k] for k in self.metro_names])
        self.metro_xy = m[:, :2]
        self.metro_p = m[:, 2] / m[:, 2].sum()
        self.segments = list(cfg["segments"])
        self.segment_p = np.array([cfg["segments"][s]["share"] for s in self.segments])
        self.days = cfg["days"]

    # ------------------------------------------------------------------ helpers
    def point_near(self, xy, deg_sd):
        return (float(xy[0] + self.rng.normal(0, deg_sd)), float(xy[1] + self.rng.normal(0, deg_sd)))

    def random_metro_point(self, exclude=None, deg_sd=0.08):
        while True:
            i = self.rng.choice(len(self.metro_names), p=self.metro_p)
            if i != exclude:
                return self.point_near(self.metro_xy[i], deg_sd), i

    def round_amount(self, x):
        p = self.cfg["product"]
        step = 100 if self.rng.random() < p["p_round_100"] else p["amount_step"]
        return max(p["amount_step"], round(x / step) * step)

    def hour_for(self, c):
        b = self.cfg["behaviour"]
        return float(np.clip(self.rng.normal(c["pref_hour"], b["hour_sd_within"]), 5, 23.5))

    # ------------------------------------------------------------------ customers
    def make_customer(self, cid):
        cfg, rng = self.cfg, self.rng
        b = cfg["behaviour"]
        seg = self.segments[rng.choice(len(self.segments), p=self.segment_p)]
        s = cfg["segments"][seg]
        home, metro_i = self.random_metro_point(deg_sd=0.12)
        late = rng.random() < b["p_late_joiner"]
        start_day = int(rng.integers(1, 90)) if late else 0
        if late:
            account_age_at_start = -start_day                        # opened during the window
        else:
            account_age_at_start = int(np.clip(rng.lognormal(np.log(1200), 0.9), 30, 12000))
        recipients = []
        for _ in range(1 + rng.poisson(b["mean_extra_recipients"])):
            if rng.random() < b["p_recipient_other_city"]:
                loc, _ = self.random_metro_point(exclude=metro_i)
            else:
                loc = self.point_near(home, 0.08)
            recipients.append((f"P{self.ids.next('recipient')}", loc))
        rate = min(s["rate"] * rng.lognormal(0, b["rate_sigma"]), b["max_monthly_rate"])
        return {
            "customer_id": cid,
            "segment": seg,
            "home_metro": self.metro_names[metro_i],
            "metro_i": metro_i,
            "home": home,
            "start_day": start_day,
            "account_age_at_start": account_age_at_start,
            "monthly_rate": rate,
            "typical_amount": s["amount"] * rng.lognormal(0, b["amount_sigma"]),
            "pref_hour": float(np.clip(rng.normal(b["pref_hour_mean"], b["pref_hour_sd"]), 6, 22)),
            "p_known": float(rng.beta(*b["p_known_beta"])),
            "p_self": float(rng.beta(*b["p_self_send_beta"])),
            "bg_rate": s["sessions"] * b["bg_sessions_scale"] * rng.lognormal(0, 0.5),
            "recipients": recipients,
            "habitual": rate >= 1.0,
            "start_limit": cfg["product"]["max_daily_limit"] if rng.random() < cfg["product"]["p_start_raised_limit"]
            else cfg["product"]["default_daily_limit"],
        }

    # ------------------------------------------------------------------ session specs
    def day_multiplier(self, day):
        b = self.cfg["behaviour"]
        date = self.start + pd.Timedelta(days=day)
        m = 1.0
        if date.day >= 25 or date.day == 1:
            m *= b["payday_boost"]
        if date.weekday() >= 5:
            m *= b["weekend_boost"]
        if date.month == 12:
            m *= b["december_boost"]
        return m

    def legit_recipient(self, c):
        """Pick a recipient for a normal voucher: self, a regular recipient, or someone new."""
        r = self.rng.random()
        if r < c["p_self"]:
            return SELF, self.point_near(c["home"], 0.03)
        if self.rng.random() < c["p_known"]:
            w = 1 / np.arange(1, len(c["recipients"]) + 1)             # first recipient is the favourite
            rid, loc = c["recipients"][self.rng.choice(len(c["recipients"]), p=w / w.sum())]
            return rid, loc
        if self.rng.random() < self.cfg["behaviour"]["p_recipient_other_city"]:
            loc, _ = self.random_metro_point(exclude=c["metro_i"])
        else:
            loc = self.point_near(c["home"], 0.08)
        return f"P{self.ids.next('recipient')}", loc

    def legit_redemption(self, loc, fast_p=None, median=None):
        cfg, rng = self.cfg["redemption"], self.rng
        if rng.random() < cfg["p_never_redeemed"]:
            return None
        if rng.random() < (cfg["p_legit_fast"] if fast_p is None else fast_p):
            delay = rng.uniform(3, 25)
        else:
            delay = rng.lognormal(np.log(median or cfg["legit_median_min"]), cfg["legit_sigma"])
        if delay > self.cfg["product"]["voucher_expiry_days"] * MIN_PER_DAY:
            return None
        channel = "ATM" if rng.random() < cfg["p_atm"] else "RETAIL"
        return delay, self.point_near(loc, 0.02), channel

    def legit_session_secs(self, c):
        s = self.cfg["session"]
        med = s["habitual_median_s"] if c["habitual"] else s["novice_median_s"]
        return float(self.rng.lognormal(np.log(med), s["sigma"]))

    def voucher_spec(self, c, day, scenario="normal", episode=0, **kw):
        rid, loc = kw.pop("recipient", None) or self.legit_recipient(c)
        amount = kw.pop("amount", None) or c["typical_amount"] * self.rng.lognormal(0, self.cfg["behaviour"]["within_amount_sigma"])
        redemption = kw.pop("redemption", "default")
        if redemption == "default":
            redemption = self.legit_redemption(loc)
        ses = self.cfg["session"]
        vouchers = [(amount, rid, redemption)]
        secs = self.legit_session_secs(c)
        if self.rng.random() < ses["p_browse"]:
            screens = 1 + self.rng.poisson(ses["browse_screens_mean"])
            secs *= self.rng.uniform(2, 5)
        else:
            screens = 1 + self.rng.poisson(ses["screens_mean"] - 1)
        if scenario == "normal" and self.rng.random() < ses["p_multi_voucher"]:
            rid2, loc2 = self.legit_recipient(c)
            if rid2 != rid:
                amt2 = c["typical_amount"] * self.rng.lognormal(0, self.cfg["behaviour"]["within_amount_sigma"])
                vouchers.append((amt2, rid2, self.legit_redemption(loc2)))
        return {
            "t": day * MIN_PER_DAY + kw.pop("hour", self.hour_for(c)) * 60,
            "kind": "voucher", "scenario": scenario, "episode": episode,
            "secs": secs, "screens": screens,
            "pre": kw.pop("pre", []),
            "vouchers": vouchers,
        }

    def normal_specs(self, c):
        specs = []
        b = self.cfg["behaviour"]
        for day in range(c["start_day"], self.days):
            lam = c["monthly_rate"] / 30 * self.day_multiplier(day)
            for _ in range(self.rng.poisson(lam)):
                specs.append(self.voucher_spec(c, day))
            for _ in range(self.rng.poisson(c["bg_rate"] / 30)):
                pre = []
                if self.rng.random() < b["p_device_change_per_month"] / max(c["bg_rate"], 1):
                    pre.append(("device_registered",))
                if self.rng.random() < b["p_password_reset_per_month"] / max(c["bg_rate"], 1):
                    pre.append(("password_reset",))
                if self.rng.random() < b["p_legit_limit_raise_per_month"] / max(c["bg_rate"], 1):
                    pre.append(("change_limit", self.cfg["product"]["max_daily_limit"]))
                specs.append({"t": day * MIN_PER_DAY + self.hour_for(c) * 60, "kind": "bg", "pre": pre})
        return specs

    def legit_episode_specs(self, c):
        cfg, rng = self.cfg, self.rng
        specs = []
        first = c["start_day"] + 1
        if first >= self.days:
            return specs
        if rng.random() < cfg["legit_episodes"]["emergency"]:
            ep = self.ids.next("episode")
            day = int(rng.integers(first, self.days))
            pre = [("change_limit", cfg["product"]["max_daily_limit"])] if rng.random() < 0.6 else []
            if rng.random() < 0.6:
                recipient = c["recipients"][0]
            else:
                loc, _ = self.random_metro_point(exclude=c["metro_i"]) if rng.random() < 0.5 else (self.point_near(c["home"], 0.08), None)
                recipient = (f"P{self.ids.next('recipient')}", loc)
            specs.append(self.voucher_spec(
                c, day, "legit_emergency", ep, recipient=recipient,
                amount=rng.uniform(0.6, 1.0) * cfg["product"]["max_daily_limit"],   # capped by the actual limit later
                hour=float(rng.uniform(6, 23)), pre=pre,
                redemption=self.legit_redemption(recipient[1], fast_p=0.5, median=60)))
        if rng.random() < cfg["legit_episodes"]["new_phone"]:
            ep = self.ids.next("episode")
            day = int(rng.integers(first, self.days))
            specs.append({"t": day * MIN_PER_DAY + self.hour_for(c) * 60, "kind": "bg",
                          "pre": [("device_registered",)]})
            day2 = min(self.days - 1, day + int(rng.integers(0, 4)))
            s = self.voucher_spec(c, day2, "legit_new_phone", ep)
            s["t"] = max(s["t"], day * MIN_PER_DAY + self.hour_for(c) * 60 + 30)
            specs.append(s)
        if rng.random() < cfg["legit_episodes"]["travel"]:
            ep = self.ids.next("episode")
            day = int(rng.integers(first, self.days))
            rid, _ = c["recipients"][0]
            far, _ = self.random_metro_point(exclude=c["metro_i"])
            specs.append(self.voucher_spec(c, day, "legit_travel", ep, recipient=(rid, far)))
        return specs

    def scam_specs(self, c):
        cfg, rng = self.cfg, self.rng
        sc = cfg["scam"]
        first = max(c["start_day"] + 1, cfg["warmup_days"])
        if first >= self.days or rng.random() >= sc["p_targeted"]:
            return []
        types = list(sc["type_weights"])
        w = np.array([sc["type_weights"][t] for t in types])
        stype = types[rng.choice(len(types), p=w / w.sum())]
        flags = {k: rng.random() < p for k, p in sc["flags"][stype].items()}
        ep = self.ids.next("episode")
        day = int(rng.integers(first, self.days))
        hour = float(rng.uniform(20, 23.5)) if rng.random() < sc["p_night"] else float(rng.uniform(8, 19))

        coached = rng.random() < sc["p_coached"]
        secs = rng.lognormal(np.log(sc["coached_median_s"] if coached else sc["rushed_median_s"]), 0.5)
        screens = int(rng.integers(6, 16)) if coached else int(rng.integers(1, 4))

        pre = []
        if rng.random() < sc["p_device_change"]:
            pre.append(("device_registered",))
        if flags["limit_raise"]:
            pre.append(("change_limit", cfg["product"]["max_daily_limit"]))
        limit_guess = cfg["product"]["max_daily_limit"] if flags["limit_raise"] else c["_limit_hint"]

        # Fraudster cash-out point: another metro, or a random spot in the victim's metro
        if rng.random() < sc["p_far_redeem"]:
            cash_loc, _ = self.random_metro_point(exclude=c["metro_i"])
        else:
            cash_loc = self.point_near(self.metro_xy[c["metro_i"]], 0.1)

        def redemption():
            if flags["fast_redeem"]:
                delay = rng.lognormal(np.log(sc["fast_redeem_median_min"]), sc["fast_redeem_sigma"])
            else:
                delay = rng.lognormal(np.log(sc["slow_redeem_median_min"]), 0.8)
            return delay, self.point_near(cash_loc, 0.01), "ATM" if rng.random() < 0.9 else "RETAIL"

        if stype == "scam_self_send" or not flags["new_recipient"]:
            rid = SELF
        else:
            rid = f"M{self.ids.next('recipient')}"          # mule / fraudster number

        vouchers = []
        if stype == "scam_split":
            for _ in range(int(rng.integers(2, 5))):
                vouchers.append((rng.beta(3, 7) * limit_guess, rid, redemption()))
        else:
            if flags["high_amount"]:
                amount = rng.beta(8, 1.5) * limit_guess          # mostly close to the limit
            elif stype == "scam_subtle":
                amount = c["typical_amount"] * rng.lognormal(np.log(2.0), 0.3)
            else:
                amount = c["typical_amount"] * rng.lognormal(np.log(1.6), 0.4)
            vouchers.append((amount, rid, redemption()))
        return [{"t": day * MIN_PER_DAY + hour * 60, "kind": "voucher", "scenario": stype, "episode": ep,
                 "secs": float(secs), "screens": screens, "pre": pre, "vouchers": vouchers}]

    # ------------------------------------------------------------------ rendering
    def render(self, c, specs, events, truth):
        """Turn session specs into ordered events while enforcing product limits."""
        rng, p = self.rng, self.cfg["product"]
        end = self.days * MIN_PER_DAY
        cid = c["customer_id"]
        limit = c["start_limit"]
        device = f"D{self.ids.next('device')}"
        spent_day, spent_month = {}, {}
        used_recipients = set()

        def ev(t, sid, etype, voucher_id="", recipient_id="", amount="", new_limit="", lat="", lon="", channel=""):
            events.append((cid, sid, round(t, 3), etype, "" if etype == "redeem_voucher" else device, voucher_id, recipient_id, amount, new_limit,
                           lat if lat == "" else round(lat, 5), lon if lon == "" else round(lon, 5), channel))

        for s in sorted(specs, key=lambda x: x["t"]):
            t = s["t"]
            if t >= end:
                continue
            sid = self.ids.next("session")
            ev(t, sid, "login")
            t += rng.uniform(3, 15) / 60
            if s["kind"] == "bg":
                ev(t, sid, "view")
            for action in s["pre"]:
                t += rng.uniform(20, 90) / 60
                if action[0] == "device_registered":
                    device = f"D{self.ids.next('device')}"
                    ev(t, sid, "device_registered")
                elif action[0] == "password_reset":
                    ev(t, sid, "password_reset")
                elif action[0] == "change_limit" and action[1] != limit:
                    limit = action[1]
                    ev(t, sid, "change_limit", new_limit=limit)
            if s["kind"] == "voucher":
                login_t = s["t"]
                n_screens = max(0, s["screens"] - 1)
                for k in range(n_screens):
                    ev(login_t + (k + 1) * s["secs"] / (n_screens + 2) / 60, sid, "view")
                t = max(t, login_t + s["secs"] / 60)
                added = set()
                for i, (amount, rid, red) in enumerate(s["vouchers"]):
                    if i > 0:
                        t += rng.uniform(40, 150) / 60
                    day, month = int(t // MIN_PER_DAY), (self.start + pd.Timedelta(minutes=t)).month
                    room = min(limit - spent_day.get(day, 0), p["monthly_limit"] - spent_month.get(month, 0))
                    amount = self.round_amount(amount)
                    if (amount > room and limit < p["max_daily_limit"] and not s["scenario"].startswith("scam")
                            and rng.random() < p["p_raise_on_demand"]):
                        # legitimate customer raises their limit to fit a larger voucher
                        limit = p["max_daily_limit"]
                        ev(t - rng.uniform(30, 90) / 60, sid, "change_limit", new_limit=limit)
                        room = min(limit - spent_day.get(day, 0), p["monthly_limit"] - spent_month.get(month, 0))
                    amount = min(amount, room - room % p["amount_step"])
                    if amount < p["amount_step"]:
                        continue                                    # limit reached: voucher not created
                    if rid != SELF and rid not in used_recipients and rid not in added:
                        ev(t - rng.uniform(15, 45) / 60, sid, "add_recipient", recipient_id=rid)
                        added.add(rid)
                    vid = self.ids.next("voucher")
                    ev(t, sid, "create_voucher", voucher_id=vid, recipient_id=rid, amount=amount)
                    spent_day[day] = spent_day.get(day, 0) + amount
                    spent_month[month] = spent_month.get(month, 0) + amount
                    used_recipients.add(rid)
                    truth.append((vid, cid, int(s["scenario"].startswith("scam")), s["scenario"], s["episode"]))
                    if red is not None:
                        delay, loc, channel = red
                        if t + delay < end:
                            ev(t + delay, sid, "redeem_voucher", voucher_id=vid, recipient_id=rid,
                               amount=amount, lat=loc[0], lon=loc[1], channel=channel)
            ev(t + rng.uniform(10, 60) / 60, sid, "logout")

    # ------------------------------------------------------------------ main loop
    def run(self, out_dir, chunk=2000):
        os.makedirs(out_dir, exist_ok=True)
        self.start = pd.Timestamp(self.cfg["start_date"])
        paths = {k: os.path.join(out_dir, f"{k}.csv") for k in ["events_log", "vouchers_truth"]}
        f_ev = open(paths["events_log"], "w", newline="")
        f_tr = open(paths["vouchers_truth"], "w", newline="")
        w_ev, w_tr = csv.writer(f_ev), csv.writer(f_tr)
        w_ev.writerow(EVENT_COLS)
        w_tr.writerow(["voucher_id", "customer_id", "is_scam", "scenario_type", "episode_id"])
        customers, latent = [], []
        events, truth = [], []
        for cid in range(1, self.cfg["n_customers"] + 1):
            c = self.make_customer(cid)
            c["_limit_hint"] = c["start_limit"]
            specs = self.normal_specs(c) + self.legit_episode_specs(c) + self.scam_specs(c)
            self.render(c, specs, events, truth)
            opened = self.start - pd.Timedelta(days=c["account_age_at_start"])
            customers.append((cid, c["home_metro"], round(c["home"][0], 5), round(c["home"][1], 5), opened.date(),
                              c["start_limit"]))
            latent.append((cid, c["segment"], round(c["monthly_rate"], 3), round(c["typical_amount"], 1),
                           round(c["pref_hour"], 2), round(c["p_known"], 3), round(c["p_self"], 3),
                           c["start_day"], c["start_limit"]))
            if cid % chunk == 0:
                events.sort(key=lambda e: (e[0], e[2]))
                w_ev.writerows(events)
                w_tr.writerows(truth)
                events, truth = [], []
        events.sort(key=lambda e: (e[0], e[2]))
        w_ev.writerows(events)
        w_tr.writerows(truth)
        f_ev.close()
        f_tr.close()
        pd.DataFrame(customers, columns=["customer_id", "home_metro", "home_lat", "home_lon", "account_open_date",
                                         "daily_limit_at_start"]) \
            .to_csv(os.path.join(out_dir, "customers.csv"), index=False)
        pd.DataFrame(latent, columns=["customer_id", "segment", "monthly_rate", "typical_amount", "pref_hour",
                                      "p_known", "p_self", "start_day", "start_limit"]) \
            .to_csv(os.path.join(out_dir, "customers_latent.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/generator.yaml")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--customers", type=int, help="override n_customers")
    ap.add_argument("--seed", type=int, help="override seed")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.customers:
        cfg["n_customers"] = a.customers
    if a.seed is not None:
        cfg["seed"] = a.seed
    t0 = time.time()
    Simulator(cfg).run(a.out)
    print(f"simulated {cfg['n_customers']} customers in {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
