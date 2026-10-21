"""Step 3: plan unusual legitimate activity and scam episodes.

Probabilities and distributions reproduce the original experiment. A branch
probability is not a guarantee that the corresponding detection rule will fire:
amounts can be capped, recipients may have no history, and delays vary.
"""
import numpy as np

from .helpers import (
    MINUTES_PER_DAY, SELF, choose_city, nearby_location, next_id,
    positive_value, usual_session_hour,
)
from .normal_activity import plan_voucher_session, plan_legitimate_redemption


def plan_legitimate_episodes(customer, config, rng, counters):
    sessions = []
    first_day = customer["start_day"] + 1
    if first_day >= config["days"]:
        return sessions
    settings = config["legit_episodes"]
    maximum_limit = config["product"]["max_daily_limit"]

    # Emergency: a large legitimate payment, possibly following a limit increase.
    if rng.random() < settings["emergency"]:
        episode_id = next_id(counters, "episode")
        day = int(rng.integers(first_day, config["days"]))
        actions = []
        if rng.random() < 0.6:
            actions.append(("change_limit", maximum_limit))

        if rng.random() < 0.6:
            recipient = customer["regular_recipients"][0]
        else:
            if rng.random() < 0.5:
                location, _ = choose_city(rng, config, excluded_city=customer["city_index"])
            else:
                location = nearby_location(rng, customer["home_location"], 0.08)
            recipient = (f"P{next_id(counters, 'recipient')}", location)

        amount = rng.uniform(0.6, 1.0) * maximum_limit
        hour = float(rng.uniform(6, 23))
        redemption = plan_legitimate_redemption(
            recipient[1], config, rng, fast_probability=0.5, median=60
        )
        sessions.append(plan_voucher_session(
            customer, day, config, rng, counters,
            scenario="legit_emergency", episode_id=episode_id,
            recipient=recipient, amount=amount, hour=hour,
            actions=actions, redemption=redemption,
        ))

    # New phone: register a device, then plan ordinary voucher use soon afterwards.
    if rng.random() < settings["new_phone"]:
        episode_id = next_id(counters, "episode")
        day = int(rng.integers(first_day, config["days"]))
        registration_hour = usual_session_hour(rng, config, customer)
        sessions.append({
            "start_minute": day * MINUTES_PER_DAY + registration_hour * 60,
            "session_type": "background",
            "account_actions": [("device_registered",)],
        })
        voucher_day = min(config["days"] - 1, day + int(rng.integers(0, 4)))
        voucher_session = plan_voucher_session(
            customer, voucher_day, config, rng, counters,
            scenario="legit_new_phone", episode_id=episode_id,
        )
        # Preserve the original second hour draw used to bound the session time.
        earliest_voucher_time = (
            day * MINUTES_PER_DAY + usual_session_hour(rng, config, customer) * 60 + 30
        )
        voucher_session["start_minute"] = max(
            voucher_session["start_minute"], earliest_voucher_time
        )
        sessions.append(voucher_session)

    # Travel: use a regular recipient, with redemption in a different city.
    if rng.random() < settings["travel"]:
        episode_id = next_id(counters, "episode")
        day = int(rng.integers(first_day, config["days"]))
        recipient_id, _ = customer["regular_recipients"][0]
        location, _ = choose_city(rng, config, excluded_city=customer["city_index"])
        sessions.append(plan_voucher_session(
            customer, day, config, rng, counters,
            scenario="legit_travel", episode_id=episode_id,
            recipient=(recipient_id, location),
        ))
    return sessions


def plan_scam_redemption(flags, location, config, rng):
    settings = config["scam"]
    if flags["fast_redeem"]:
        delay = positive_value(
            rng, settings["fast_redeem_median_min"], settings["fast_redeem_sigma"]
        )
    else:
        delay = positive_value(rng, settings["slow_redeem_median_min"], 0.8)
    redemption_location = nearby_location(rng, location, 0.01)
    if rng.random() < 0.9:
        channel = "ATM"
    else:
        channel = "RETAIL"
    # This original branch has no explicit expiry-discard check. Keep it unchanged.
    return delay, redemption_location, channel


def plan_scam_episode(customer, config, rng, counters):
    settings = config["scam"]
    first_day = max(customer["start_day"] + 1, config["warmup_days"])
    if first_day >= config["days"]:
        return []
    if rng.random() >= settings["p_targeted"]:
        return []  # This customer does not receive a planned scam episode.

    # 1. Choose a scenario, then draw its warning-pattern flags independently.
    scenario_names = list(settings["type_weights"])
    weights = np.array([settings["type_weights"][name] for name in scenario_names])
    scenario_index = rng.choice(len(scenario_names), p=weights / weights.sum())
    scenario = scenario_names[scenario_index]
    flags = {}
    for name, probability in settings["flags"][scenario].items():
        flags[name] = rng.random() < probability

    episode_id = next_id(counters, "episode")
    day = int(rng.integers(first_day, config["days"]))
    if rng.random() < settings["p_night"]:
        hour = float(rng.uniform(20, 23.5))
    else:
        hour = float(rng.uniform(8, 19))

    # 2. A coached session usually lasts longer and specifies more screens.
    coached = rng.random() < settings["p_coached"]
    if coached:
        duration_median = settings["coached_median_s"]
    else:
        duration_median = settings["rushed_median_s"]
    duration_seconds = positive_value(rng, duration_median, 0.5)
    if coached:
        screen_count = int(rng.integers(6, 16))  # 6 through 15
    else:
        screen_count = int(rng.integers(1, 4))   # 1 through 3

    # 3. Plan any account changes before voucher creation.
    actions = []
    if rng.random() < settings["p_device_change"]:
        actions.append(("device_registered",))
    if flags["limit_raise"]:
        actions.append(("change_limit", config["product"]["max_daily_limit"]))
        assumed_limit = config["product"]["max_daily_limit"]
    else:
        assumed_limit = customer["starting_limit"]

    # 4. Pick a proposed redemption location.
    if rng.random() < settings["p_far_redeem"]:
        location, _ = choose_city(rng, config, excluded_city=customer["city_index"])
    else:
        city_name = list(config["metros"])[customer["city_index"]]
        city_centre = np.array(config["metros"][city_name][:2])
        location = nearby_location(rng, city_centre, 0.1)

    # 5. Choose recipient. In the original, a false new-recipient flag means SELF.
    if scenario == "scam_self_send" or not flags["new_recipient"]:
        recipient_id = SELF
    else:
        recipient_id = f"M{next_id(counters, 'recipient')}"

    # 6. Build proposed vouchers. events.py will apply actual sending limits.
    vouchers = []
    if scenario == "scam_split":
        voucher_count = int(rng.integers(2, 5))  # attempts 2 through 4
        for _ in range(voucher_count):
            amount = rng.beta(3, 7) * assumed_limit
            redemption = plan_scam_redemption(flags, location, config, rng)
            vouchers.append({"amount": amount, "recipient_id": recipient_id, "redemption": redemption})
    else:
        if flags["high_amount"]:
            amount = rng.beta(8, 1.5) * assumed_limit
        elif scenario == "scam_subtle":
            amount = customer["typical_amount"] * positive_value(rng, 2.0, 0.3)
        else:
            amount = customer["typical_amount"] * positive_value(rng, 1.6, 0.4)
        redemption = plan_scam_redemption(flags, location, config, rng)
        vouchers.append({"amount": amount, "recipient_id": recipient_id, "redemption": redemption})

    return [{
        "start_minute": day * MINUTES_PER_DAY + hour * 60,
        "session_type": "voucher",
        "scenario_type": scenario,
        "episode_id": episode_id,
        "duration_seconds": float(duration_seconds),
        "screen_count": screen_count,
        "account_actions": actions,
        "vouchers": vouchers,
    }]
