"""Step 1: create one fictional customer and their individual habits.

These habits drive the generation process. They are not model inputs.
The ordinary customer CSV and the hidden-habits CSV are written by run.py.
"""
import numpy as np

from .helpers import choose_city, nearby_location, next_id, positive_value


def create_customer(customer_id, config, rng, counters):
    behaviour = config["behaviour"]
    product = config["product"]

    # 1. Select a customer group using the shares in generator.yaml.
    group_names = list(config["segments"])
    group_shares = np.array([
        config["segments"][name]["share"] for name in group_names
    ])
    group_index = rng.choice(len(group_names), p=group_shares)
    group_name = group_names[group_index]
    group = config["segments"][group_name]

    # 2. Give the customer a home location.
    home_location, city_index = choose_city(rng, config, spread=0.12)
    city_names = list(config["metros"])

    # 3. Some accounts open during the observation window.
    joins_later = rng.random() < behaviour["p_late_joiner"]
    if joins_later:
        start_day = int(rng.integers(1, 90))  # 1 to 89 inclusive
        account_age_at_start = -start_day
    else:
        start_day = 0
        sampled_age = positive_value(rng, median=1200, spread=0.9)
        account_age_at_start = int(np.clip(sampled_age, 30, 12000))

    # 4. Create a small list of regular recipients.
    # They become 'known' in the extracted history only after actual voucher use.
    regular_recipients = []
    recipient_count = 1 + rng.poisson(behaviour["mean_extra_recipients"])
    for _ in range(recipient_count):
        if rng.random() < behaviour["p_recipient_other_city"]:
            location, _ = choose_city(rng, config, excluded_city=city_index)
        else:
            location = nearby_location(rng, home_location, spread=0.08)
        recipient_id = f"P{next_id(counters, 'recipient')}"
        regular_recipients.append((recipient_id, location))

    # 5. Vary habits between customers, even within the same group.
    rate_multiplier = rng.lognormal(0, behaviour["rate_sigma"])
    monthly_rate = min(group["rate"] * rate_multiplier, behaviour["max_monthly_rate"])
    typical_amount = group["amount"] * rng.lognormal(0, behaviour["amount_sigma"])
    preferred_hour = float(np.clip(
        rng.normal(behaviour["pref_hour_mean"], behaviour["pref_hour_sd"]), 6, 22
    ))

    # Beta draws produce customer-specific probabilities between 0 and 1.
    known_recipient_probability = float(rng.beta(*behaviour["p_known_beta"]))
    self_send_probability = float(rng.beta(*behaviour["p_self_send_beta"]))
    background_monthly_rate = (
        group["sessions"] * behaviour["bg_sessions_scale"] * rng.lognormal(0, 0.5)
    )

    # 6. Some customers already have the higher limit when the window begins.
    if rng.random() < product["p_start_raised_limit"]:
        starting_limit = product["max_daily_limit"]
    else:
        starting_limit = product["default_daily_limit"]

    return {
        "customer_id": customer_id,
        "segment": group_name,
        "home_city": city_names[city_index],
        "city_index": city_index,
        "home_location": home_location,
        "start_day": start_day,
        "account_age_at_start": account_age_at_start,
        "monthly_rate": monthly_rate,
        "typical_amount": typical_amount,
        "preferred_hour": preferred_hour,
        "known_recipient_probability": known_recipient_probability,
        "self_send_probability": self_send_probability,
        "background_monthly_rate": background_monthly_rate,
        "regular_recipients": regular_recipients,
        "is_habitual": monthly_rate >= 1.0,
        "starting_limit": starting_limit,
    }
