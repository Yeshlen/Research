"""Start here: generate customers, plan their activity, record it and save four CSVs."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .customers import create_customer
from .normal_activity import plan_ordinary_activity
from .scenarios import plan_legitimate_episodes, plan_scam_episode
from .events import record_customer_events
from .helpers import EVENT_COLUMNS


def generate_dataset(config, output_folder, chunk_size=2000):
    """Use the same random process and CSV format as the original archived/simulate.py.

    Batches limit memory use. The feature CSV is created separately by features.py.
    Random calls stay in their original order so seeded runs can be compared.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    counters = {"session": 0, "voucher": 0, "recipient": 0, "device": 0, "episode": 0}
    start_date = pd.Timestamp(config["start_date"])
    customer_rows = []
    habit_rows = []
    event_rows = []
    truth_rows = []

    events_file = output_folder / "events_log.csv"
    truth_file = output_folder / "vouchers_truth.csv"
    with events_file.open("w", newline="") as events_stream, truth_file.open("w", newline="") as truth_stream:
        events_writer = csv.writer(events_stream)
        truth_writer = csv.writer(truth_stream)
        events_writer.writerow(EVENT_COLUMNS)
        truth_writer.writerow(["voucher_id", "customer_id", "is_scam", "scenario_type", "episode_id"])

        for customer_id in range(1, config["n_customers"] + 1):
            # STEP 1: Give one customer a profile and their own habits.
            customer = create_customer(customer_id, config, rng, counters)

            # STEP 2: Plan ordinary sessions, legitimate episodes, then any scam.
            sessions = plan_ordinary_activity(customer, config, rng, counters, start_date)
            sessions += plan_legitimate_episodes(customer, config, rng, counters)
            sessions += plan_scam_episode(customer, config, rng, counters)

            # STEP 3: Apply limits and turn the plans into events and truth labels.
            record_customer_events(
                customer, sessions, config, rng, counters,
                event_rows, truth_rows, start_date,
            )

            # STEP 4: Save the public profile separately from hidden generation habits.
            opened = start_date - pd.Timedelta(days=customer["account_age_at_start"])
            customer_rows.append((
                customer_id, customer["home_city"],
                round(customer["home_location"][0], 5),
                round(customer["home_location"][1], 5),
                opened.date(), customer["starting_limit"],
            ))
            habit_rows.append((
                customer_id, customer["segment"], round(customer["monthly_rate"], 3),
                round(customer["typical_amount"], 1), round(customer["preferred_hour"], 2),
                round(customer["known_recipient_probability"], 3),
                round(customer["self_send_probability"], 3),
                customer["start_day"], customer["starting_limit"],
            ))

            if customer_id % chunk_size == 0:
                write_event_batch(events_writer, truth_writer, event_rows, truth_rows)
                event_rows = []
                truth_rows = []

        write_event_batch(events_writer, truth_writer, event_rows, truth_rows)

    # Keep original column names so the unchanged feature extractor can read them.
    customer_columns = [
        "customer_id", "home_metro", "home_lat", "home_lon", "account_open_date",
        "daily_limit_at_start",
    ]
    habit_columns = [
        "customer_id", "segment", "monthly_rate", "typical_amount", "pref_hour",
        "p_known", "p_self", "start_day", "start_limit",
    ]
    pd.DataFrame(customer_rows, columns=customer_columns).to_csv(
        output_folder / "customers.csv", index=False
    )
    pd.DataFrame(habit_rows, columns=habit_columns).to_csv(
        output_folder / "customers_latent.csv", index=False
    )


def write_event_batch(events_writer, truth_writer, event_rows, truth_rows):
    """Sort by customer and minute before writing this batch to disk."""
    event_rows.sort(key=lambda event: (event[0], event[2]))
    events_writer.writerows(event_rows)
    truth_writer.writerows(truth_rows)
