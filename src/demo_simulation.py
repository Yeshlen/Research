"""Independent dashboard examples using the existing session planners and feature extractor.

Original data is read only. Each call branches from one customer's earlier history,
then saves only the newly generated features and events under data/demo/.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import json
import os

import numpy as np
import pandas as pd
import yaml

from features import extract
from simulator.events import record_customer_events
from simulator.helpers import EVENT_COLUMNS, MINUTES_PER_DAY, SELF, nearby_location
from simulator.normal_activity import plan_voucher_session
from simulator.scenarios import plan_legitimate_episodes, plan_scam_episode

LEGIT_SCENARIOS = ["normal", "legit_emergency", "legit_new_phone", "legit_travel"]
SCAM_SCENARIOS = ["scam_classic", "scam_self_send", "scam_limit_raise", "scam_split", "scam_subtle"]
TRUTH_COLUMNS = ["voucher_id", "customer_id", "is_scam", "scenario_type", "episode_id"]


def generation_config(root):
    """Use the settings of the existing dataset, rather than a subsequently edited YAML."""
    root = Path(root)
    metadata = root / "data/synthetic/dataset_metadata.json"
    if metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved.get("generation_config"):
            return deepcopy(saved["generation_config"])
    return yaml.safe_load((root / "config/generator.yaml").read_text())


def customer_history(root, customer_id, cutoff_minute):
    """Read one customer without loading the whole event log into memory."""
    parts = []
    for chunk in pd.read_csv(Path(root) / "data/synthetic/events_log.csv", chunksize=200000,
                             dtype={"recipient_id": str, "device_id": str, "channel": str}):
        found = chunk[(chunk.customer_id == customer_id) & (chunk.t_min < cutoff_minute)]
        if not found.empty:
            parts.append(found)
    if not parts:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values("t_min", kind="stable")


def restore_customer(profile, habits, history, config, rng):
    """Recover observable state; original hidden recipient lists were not saved."""
    home = (float(profile.home_lat), float(profile.home_lon))
    creations = history[history.event_type == "create_voucher"].copy()
    redemptions = history[history.event_type == "redeem_voucher"]
    regular = []
    for recipient in creations.recipient_id.dropna().value_counts().index:
        if recipient == SELF:
            continue
        past = redemptions[redemptions.recipient_id == recipient]
        location = home if past.empty else (float(past.iloc[-1].lat), float(past.iloc[-1].lon))
        regular.append((str(recipient), location))
    if not regular:
        # No observed recipient: create a fictional regular recipient, still unknown in history.
        regular = [(f"P_demo_{int(rng.integers(10**9))}", nearby_location(rng, home, 0.08))]

    changes = history[history.event_type == "change_limit"]
    limit = float(profile.daily_limit_at_start if changes.empty else changes.iloc[-1].new_limit)
    city_names = list(config["metros"])
    if profile.home_metro not in city_names:
        raise ValueError("The customer city does not match the recorded generation settings.")
    customer = dict(customer_id=int(profile.customer_id), home_location=home,
                    city_index=city_names.index(profile.home_metro),
                    start_day=int(habits.start_day), starting_limit=limit,
                    monthly_rate=float(habits.monthly_rate), typical_amount=float(habits.typical_amount),
                    preferred_hour=float(habits.pref_hour),
                    known_recipient_probability=float(habits.p_known),
                    self_send_probability=float(habits.p_self), regular_recipients=regular,
                    is_habitual=float(habits.monthly_rate) >= 1)
    if creations.empty:
        daily, monthly = {}, {}
    else:
        creations["calendar_day"] = (creations.t_min // MINUTES_PER_DAY).astype(int)
        timestamps = pd.Timestamp(config["start_date"]) + pd.to_timedelta(creations.t_min, unit="min")
        creations["calendar_month"] = timestamps.dt.month
        daily = creations.groupby("calendar_day").amount.sum().to_dict()
        monthly = creations.groupby("calendar_month").amount.sum().to_dict()
    devices = history.device_id.dropna()
    state = dict(current_limit=limit, daily_spending=daily, monthly_spending=monthly,
                 used_recipients=set(creations.recipient_id.dropna()))
    if not devices.empty:
        state["device_id"] = str(devices.iloc[-1])
    return customer, state


def plan_example(customer, kind, scenario, cutoff_day, config, rng, counters):
    """Reuse original planners, conditioning on the class/scenario requested by the user."""
    planner_customer = deepcopy(customer)
    planner_customer["start_day"] = cutoff_day - 1
    settings = deepcopy(config)
    if kind == "Scam":
        settings["scam"]["p_targeted"] = 1.0
        if scenario != "Automatic":
            if scenario not in SCAM_SCENARIOS:
                raise ValueError("Choose a scam scenario for a scam example.")
            settings["scam"]["type_weights"] = {scenario: 1.0}
        return plan_scam_episode(planner_customer, settings, rng, counters)
    if scenario in ("Automatic", "normal"):
        return [plan_voucher_session(planner_customer, cutoff_day, settings, rng, counters)]
    if scenario not in LEGIT_SCENARIOS:
        raise ValueError("Choose a legitimate scenario for a legitimate example.")
    wanted = scenario.removeprefix("legit_")
    settings["legit_episodes"] = {name: float(name == wanted) for name in settings["legit_episodes"]}
    return plan_legitimate_episodes(planner_customer, settings, rng, counters)


def generate_example(root, kind="Random", scenario="Automatic", customer_id=None,
                     cutoff_day=None, seed=None):
    """Generate one independent session/episode, possibly containing several vouchers.

    Random is 50/50 class choice. Automatic legitimate means an ordinary session;
    automatic scam uses the original scenario weights. This is demonstration sampling,
    not a new sample with the research dataset's prevalence.
    """
    root = Path(root)
    config = generation_config(root)
    rng = np.random.default_rng(seed)
    if kind not in ("Legitimate", "Scam", "Random"):
        raise ValueError("Unknown generation type")
    if kind == "Random":
        kind = str(rng.choice(["Legitimate", "Scam"]))
    if cutoff_day is None:
        cutoff_day = int(rng.integers(120, min(166, config["days"] - 1)))
    if not 30 <= cutoff_day < config["days"] - 1:
        raise ValueError("Choose a history checkpoint within the simulation window.")

    data = root / "data/synthetic"
    customers = pd.read_csv(data / "customers.csv")
    habits = pd.read_csv(data / "customers_latent.csv")
    eligible = habits[habits.start_day < cutoff_day]
    if customer_id is None:
        customer_id = int(rng.choice(eligible.customer_id.to_numpy()))
    if customer_id not in set(eligible.customer_id):
        raise ValueError("The customer must exist before the history checkpoint.")
    profile = customers.set_index("customer_id", drop=False).loc[customer_id]
    latent = habits.set_index("customer_id").loc[customer_id]
    cutoff = cutoff_day * MINUTES_PER_DAY
    history = customer_history(root, customer_id, cutoff)
    customer, state = restore_customer(profile, latent, history, config, rng)
    offset = int(rng.integers(10**9, 10**12))
    counters = {name: offset for name in ("session", "voucher", "recipient", "device", "episode")}
    sessions = plan_example(customer, kind, scenario, cutoff_day, config, rng, counters)
    event_rows, truth_rows = [], []
    record_customer_events(customer, sessions, config, rng, counters, event_rows, truth_rows,
                           pd.Timestamp(config["start_date"]), initial_state=state)
    if not truth_rows:
        raise ValueError("No voucher could be created within this customer's allowance. Generate another example.")
    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS).sort_values("t_min", kind="stable")
    if (events.t_min < cutoff).any():
        raise ValueError("A planned event preceded the independent history checkpoint.")
    truth = pd.DataFrame(truth_rows, columns=TRUTH_COLUMNS)

    # Reuse the research extractor unchanged, on a temporary copy of just this customer.
    with TemporaryDirectory(prefix="cardless-demo-") as folder:
        temp = Path(folder)
        combined = pd.concat([history, events], ignore_index=True).sort_values("t_min", kind="stable")
        combined.to_csv(temp / "events_log.csv", index=False)
        pd.DataFrame([profile]).to_csv(temp / "customers.csv", index=False)
        truth.to_csv(temp / "vouchers_truth.csv", index=False)
        all_features = extract(str(temp), config)
    features = all_features[all_features.voucher_id.isin(truth.voucher_id)].copy()
    if len(features) != len(truth) or features.is_scam.isna().any():
        raise ValueError("Generated features and scenario labels did not match.")
    expected = int(kind == "Scam")
    if not (features.is_scam == expected).all():
        raise ValueError("The generator returned the wrong scenario class.")
    features["is_scam"] = features.is_scam.astype(int)
    simulation_id = uuid4().hex[:12]
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    for frame in (features, events):
        frame["simulation_id"] = simulation_id
        frame["generated_at"] = generated_at
    features["history_cutoff"] = pd.Timestamp(config["start_date"]) + pd.Timedelta(days=cutoff_day)
    features["requested_kind"] = kind
    features["split"] = "demo"
    return features.reset_index(drop=True), events.reset_index(drop=True)


def save_example(root, features, events):
    """Separate local CSVs; never append to research inputs or evaluation results."""
    folder = Path(root) / "data/demo"
    folder.mkdir(parents=True, exist_ok=True)
    vouchers_path = folder / "generated_vouchers.csv"
    if features.voucher_id.duplicated().any():
        raise ValueError("Generated voucher identifiers must be unique.")
    if vouchers_path.exists():
        existing_ids = set(pd.read_csv(vouchers_path, usecols=["voucher_id"]).voucher_id)
        if existing_ids.intersection(features.voucher_id):
            raise ValueError("This generated voucher has already been saved.")
    for filename, frame in [("generated_events.csv", events), ("generated_vouchers.csv", features)]:
        path = folder / filename
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        combined = frame.copy() if existing.empty else pd.concat([existing, frame], ignore_index=True)
        if filename == "generated_vouchers.csv" and combined.voucher_id.duplicated().any():
            raise ValueError("This generated voucher has already been saved.")
        temporary = path.with_suffix(".tmp")
        combined.to_csv(temporary, index=False)
        os.replace(temporary, path)


def load_examples(root):
    path = Path(root) / "data/demo/generated_vouchers.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["created_at", "history_cutoff"])
