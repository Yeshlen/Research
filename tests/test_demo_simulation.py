"""New demonstrations must remain independent and respect the original event semantics."""
from copy import deepcopy
from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demo_simulation import (SCAM_SCENARIOS, generate_example, load_examples, save_example)
from demo_scoring import score_vouchers
from models import STAGE_A_FEATURES, STAGE_B_FEATURES
from simulator import generate_dataset
from simulator.events import record_customer_events


@pytest.fixture
def research(tmp_path):
    config = yaml.safe_load((ROOT / "config/generator.yaml").read_text())
    config["n_customers"] = 25
    config["seed"] = 42
    (tmp_path / "config").mkdir()
    (tmp_path / "config/generator.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    generate_dataset(config, tmp_path / "data/synthetic")
    return tmp_path


@pytest.mark.parametrize("scenario", SCAM_SCENARIOS + ["normal", "legit_emergency", "legit_new_phone", "legit_travel"])
def test_requested_scenario_does_not_depend_on_detection(research, scenario):
    kind = "Scam" if scenario.startswith("scam") else "Legitimate"
    records, events = generate_example(research, kind, scenario, customer_id=1, cutoff_day=120, seed=23)
    assert set(records.scenario_type) == {scenario}
    assert set(records.is_scam) == {int(kind == "Scam")}
    assert set(records.split) == {"demo"}
    assert (records.amount > 0).all()
    assert (records.amount_to_limit_ratio <= 1).all()
    assert (events.t_min >= 120 * 1440).all()
    assert records.voucher_id.is_unique
    if scenario == "scam_split":
        assert 2 <= len(records) <= 4
        assert records.vouchers_in_session.tolist() == list(range(len(records)))
        assert records.history_count.tolist() == list(range(int(records.history_count.iloc[0]),
                                                            int(records.history_count.iloc[0]) + len(records)))


def test_independent_examples_leave_research_files_unchanged(research):
    data = research / "data/synthetic"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    one, events = generate_example(research, "Scam", "scam_split", customer_id=1, cutoff_day=120, seed=23)
    save_example(research, one, events)
    two, _ = generate_example(research, "Scam", "scam_split", customer_id=1, cutoff_day=120, seed=23)
    pd.testing.assert_frame_equal(one[STAGE_A_FEATURES + STAGE_B_FEATURES[-5:]],
                                  two[STAGE_A_FEATURES + STAGE_B_FEATURES[-5:]])
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    assert len(load_examples(research)) == len(one)


def test_future_original_events_do_not_enter_the_copied_history(research):
    one, _ = generate_example(research, "Scam", "scam_split", customer_id=1, cutoff_day=120, seed=23)
    path = research / "data/synthetic/events_log.csv"
    events = pd.read_csv(path)
    events.loc[(events.customer_id == 1) & (events.t_min >= 120 * 1440), "amount"] = 999999
    events.to_csv(path, index=False)
    two, _ = generate_example(research, "Scam", "scam_split", customer_id=1, cutoff_day=120, seed=23)
    pd.testing.assert_frame_equal(one[STAGE_A_FEATURES + STAGE_B_FEATURES[-5:]],
                                  two[STAGE_A_FEATURES + STAGE_B_FEATURES[-5:]])


def test_recorder_respects_existing_spending_and_recipient_state():
    config = yaml.safe_load((ROOT / "config/generator.yaml").read_text())
    customer = dict(customer_id=1, starting_limit=5000)
    sessions = [dict(start_minute=600, session_type="voucher", account_actions=[],
                     duration_seconds=70, screen_count=3, scenario_type="normal", episode_id=0,
                     vouchers=[dict(amount=2000, recipient_id="P1", redemption=None)])]
    state = dict(current_limit=5000, daily_spending={0: 4950}, monthly_spending={7: 24950},
                 used_recipients={"P1"}, device_id="D_old")
    before = deepcopy(state)
    events, truth = [], []
    counters = {k: 100 for k in ("device", "session", "voucher", "recipient", "episode")}
    record_customer_events(customer, sessions, config, np.random.default_rng(1), counters, events, truth,
                           pd.Timestamp("2025-07-01"), initial_state=state)
    creates = [e for e in events if e[3] == "create_voucher"]
    assert len(creates) == 1 and creates[0][7] == 50
    assert not any(e[3] == "add_recipient" for e in events)
    assert state == before


class ConstantScorer:
    def score(self, frame):
        assert not {"is_scam", "scenario_type", "customer_id", "voucher_id"} & set(frame.columns)
        return np.full(len(frame), 0.2)


def test_alert_uses_saved_threshold_not_truth_label(research):
    records, _ = generate_example(research, "Scam", "scam_split", customer_id=1, cutoff_day=120, seed=23)
    selected = dict(thresholds={"LOF": .9}, scorers={"LOF": ConstantScorer()})
    assert (records.is_scam == 1).all()
    decisions = score_vouchers(records, selected, "LOF", "A")
    assert not any(d["alert"] for d in decisions)
    changed = records.copy()
    changed["is_scam"] = 0
    assert decisions == score_vouchers(changed, selected, "LOF", "A")


def test_redemption_requires_redemption_data(research):
    records, _ = generate_example(research, "Scam", customer_id=1, cutoff_day=120, seed=23)
    records["redeemed"] = 0
    with pytest.raises(ValueError, match="redemption"):
        score_vouchers(records, {"thresholds": {"Rules": 80}}, "Rules", "B")


def test_duplicate_save_does_not_change_either_demo_file(research):
    records, events = generate_example(research, "Scam", "scam_split", customer_id=1, cutoff_day=120, seed=23)
    save_example(research, records, events)
    folder = research / "data/demo"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(ValueError, match="already been saved"):
        save_example(research, records, events)
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
