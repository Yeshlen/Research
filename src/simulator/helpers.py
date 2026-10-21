"""Small shared operations: IDs, locations and positive random values.

The main generation steps are in customers.py, normal_activity.py and scenarios.py.
You can return to these helpers when you want to understand a particular draw.
"""
import numpy as np

MINUTES_PER_DAY = 1440
SELF = "SELF"
EVENT_COLUMNS = [
    "customer_id", "session_id", "t_min", "event_type", "device_id",
    "voucher_id", "recipient_id", "amount", "new_limit", "lat", "lon", "channel",
]


def next_id(counters, kind):
    """Give the next session, voucher, recipient, device or episode a unique number."""
    counters[kind] += 1
    return counters[kind]


def positive_value(rng, median, spread):
    """Draw a positive value around a median, with some larger values in the tail.

    This is the original lognormal distribution. 'spread' is its log-scale
    standard deviation, not a hard range or a standard deviation in rand/seconds.
    """
    return rng.lognormal(np.log(median), spread)


def nearby_location(rng, location, spread):
    """Add a small random latitude/longitude offset. spread is in degrees, not km."""
    latitude = float(location[0] + rng.normal(0, spread))
    longitude = float(location[1] + rng.normal(0, spread))
    return latitude, longitude


def choose_city(rng, config, excluded_city=None, spread=0.08):
    """Choose a city using its configured weight, then choose a point near it."""
    names = list(config["metros"])
    city_settings = np.array([config["metros"][name] for name in names])
    coordinates = city_settings[:, :2]
    weights = city_settings[:, 2]
    probabilities = weights / weights.sum()

    # Keep drawing if this must be a different city from the customer's city.
    while True:
        city_index = rng.choice(len(names), p=probabilities)
        if city_index != excluded_city:
            location = nearby_location(rng, coordinates[city_index], spread)
            return location, city_index


def usual_session_hour(rng, config, customer):
    """Choose an ordinary session-start hour around this customer's preference."""
    hour = rng.normal(
        customer["preferred_hour"], config["behaviour"]["hour_sd_within"]
    )
    return float(np.clip(hour, 5, 23.5))


def round_amount(rng, config, amount):
    """Use R100 rounding on one branch and the configured R50 step otherwise."""
    product = config["product"]
    if rng.random() < product["p_round_100"]:
        step = 100
    else:
        step = product["amount_step"]
    rounded = round(amount / step) * step
    return max(product["amount_step"], rounded)
