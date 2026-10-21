"""Step 2: plan ordinary activity before recording the actual events.

A session plan is a dictionary describing its start, duration, actions and
proposed vouchers. events.py later applies limits and creates timestamped rows.
"""
import numpy as np
import pandas as pd

from .helpers import (
    MINUTES_PER_DAY, SELF, choose_city, nearby_location, next_id,
    positive_value, usual_session_hour,
)


def choose_legitimate_recipient(customer, config, rng, counters):
    # First decide whether to send to self.
    if rng.random() < customer["self_send_probability"]:
        return SELF, nearby_location(rng, customer["home_location"], 0.03)

    # Otherwise, possibly use a regular recipient. Earlier list entries are favoured.
    if rng.random() < customer["known_recipient_probability"]:
        recipients = customer["regular_recipients"]
        weights = 1 / np.arange(1, len(recipients) + 1)
        recipient_index = rng.choice(len(recipients), p=weights / weights.sum())
        return recipients[recipient_index]

    # Otherwise create a new recipient.
    if rng.random() < config["behaviour"]["p_recipient_other_city"]:
        location, _ = choose_city(rng, config, excluded_city=customer["city_index"])
    else:
        location = nearby_location(rng, customer["home_location"], 0.08)
    recipient_id = f"P{next_id(counters, 'recipient')}"
    return recipient_id, location


def plan_legitimate_redemption(location, config, rng, fast_probability=None, median=None):
    settings = config["redemption"]
    if rng.random() < settings["p_never_redeemed"]:
        return None

    if fast_probability is None:
        fast_probability = settings["p_legit_fast"]
    if rng.random() < fast_probability:
        delay_minutes = rng.uniform(3, 25)
    else:
        delay_median = median or settings["legit_median_min"]
        delay_minutes = positive_value(rng, delay_median, settings["legit_sigma"])

    expiry_minutes = config["product"]["voucher_expiry_days"] * MINUTES_PER_DAY
    if delay_minutes > expiry_minutes:
        return None

    if rng.random() < settings["p_atm"]:
        channel = "ATM"
    else:
        channel = "RETAIL"
    redemption_location = nearby_location(rng, location, 0.02)
    return delay_minutes, redemption_location, channel


def plan_voucher_session(
    customer, day, config, rng, counters, scenario="normal", episode_id=0,
    recipient=None, amount=None, redemption="default", hour=None, actions=None,
):
    """Plan one voucher session; special scenarios can override selected values."""
    if not recipient:
        recipient = choose_legitimate_recipient(customer, config, rng, counters)
    recipient_id, recipient_location = recipient

    if not amount:
        amount = customer["typical_amount"] * rng.lognormal(
            0, config["behaviour"]["within_amount_sigma"]
        )
    if redemption == "default":
        redemption = plan_legitimate_redemption(recipient_location, config, rng)

    vouchers = [{"amount": amount, "recipient_id": recipient_id, "redemption": redemption}]
    settings = config["session"]
    if customer["is_habitual"]:
        duration_median = settings["habitual_median_s"]
    else:
        duration_median = settings["novice_median_s"]
    duration_seconds = float(positive_value(rng, duration_median, settings["sigma"]))

    if rng.random() < settings["p_browse"]:
        screen_count = 1 + rng.poisson(settings["browse_screens_mean"])
        duration_seconds *= rng.uniform(2, 5)
    else:
        screen_count = 1 + rng.poisson(settings["screens_mean"] - 1)

    # An ordinary session may attempt a second voucher to someone different.
    if scenario == "normal" and rng.random() < settings["p_multi_voucher"]:
        second_recipient, second_location = choose_legitimate_recipient(
            customer, config, rng, counters
        )
        if second_recipient != recipient_id:
            second_amount = customer["typical_amount"] * rng.lognormal(
                0, config["behaviour"]["within_amount_sigma"]
            )
            second_redemption = plan_legitimate_redemption(second_location, config, rng)
            vouchers.append({
                "amount": second_amount,
                "recipient_id": second_recipient,
                "redemption": second_redemption,
            })

    # The original draws an ordinary hour even when a scenario supplies an override.
    # Retain that draw so the seed produces exactly the same later random choices.
    ordinary_hour = usual_session_hour(rng, config, customer)
    if hour is None:
        hour = ordinary_hour

    return {
        "start_minute": day * MINUTES_PER_DAY + hour * 60,
        "session_type": "voucher",
        "scenario_type": scenario,
        "episode_id": episode_id,
        "duration_seconds": duration_seconds,
        "screen_count": screen_count,
        "account_actions": [] if actions is None else actions,
        "vouchers": vouchers,
    }


def activity_multiplier(day, start_date, config):
    """Increase the expected activity on selected calendar days."""
    date = start_date + pd.Timedelta(days=day)
    settings = config["behaviour"]
    multiplier = 1.0
    if date.day >= 25 or date.day == 1:
        multiplier *= settings["payday_boost"]
    if date.weekday() >= 5:
        multiplier *= settings["weekend_boost"]
    if date.month == 12:
        multiplier *= settings["december_boost"]
    return multiplier


def plan_ordinary_activity(customer, config, rng, counters, start_date):
    sessions = []
    settings = config["behaviour"]
    for day in range(customer["start_day"], config["days"]):
        # Poisson draws choose a count: zero, one or several sessions on this day.
        expected_voucher_sessions = customer["monthly_rate"] / 30
        expected_voucher_sessions *= activity_multiplier(day, start_date, config)
        voucher_session_count = rng.poisson(expected_voucher_sessions)
        for _ in range(voucher_session_count):
            sessions.append(plan_voucher_session(customer, day, config, rng, counters))

        expected_background_sessions = customer["background_monthly_rate"] / 30
        background_session_count = rng.poisson(expected_background_sessions)
        for _ in range(background_session_count):
            actions = []
            denominator = max(customer["background_monthly_rate"], 1)
            if rng.random() < settings["p_device_change_per_month"] / denominator:
                actions.append(("device_registered",))
            if rng.random() < settings["p_password_reset_per_month"] / denominator:
                actions.append(("password_reset",))
            if rng.random() < settings["p_legit_limit_raise_per_month"] / denominator:
                actions.append(("change_limit", config["product"]["max_daily_limit"]))
            hour = usual_session_hour(rng, config, customer)
            sessions.append({
                "start_minute": day * MINUTES_PER_DAY + hour * 60,
                "session_type": "background",
                "account_actions": actions,
            })
    return sessions
