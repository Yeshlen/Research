"""The clearer implementation must preserve the original experiment's outputs."""
from pathlib import Path
import sys
import copy

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from archived.simulate import Simulator as ArchivedSimulator
from features import extract
from validate import check_dataset
from simulator import generate_dataset


@pytest.mark.parametrize("seed", [42, 7, 123, 75])
def test_same_csvs_and_features_as_original(seed, tmp_path):
    config = yaml.safe_load((ROOT / "config/generator.yaml").read_text())
    config["seed"] = seed
    config["n_customers"] = 800
    original_folder = tmp_path / "original"
    current_folder = tmp_path / "current"

    # Small batches exercise the memory-flush boundary as well as the final batch.
    ArchivedSimulator(copy.deepcopy(config)).run(str(original_folder), chunk=137)
    generate_dataset(copy.deepcopy(config), current_folder, chunk_size=137)
    for filename in ["events_log.csv", "vouchers_truth.csv", "customers.csv", "customers_latent.csv"]:
        assert (original_folder / filename).read_bytes() == (current_folder / filename).read_bytes(), filename

    original_features = extract(str(original_folder), config)
    current_features = extract(str(current_folder), config)
    pd.testing.assert_frame_equal(original_features, current_features)
    truth = pd.read_csv(current_folder / "vouchers_truth.csv")
    checks = check_dataset(current_features, truth, config)
    assert len(checks) == 20
    assert checks.violations.sum() == 0
    assert set(config["scam"]["type_weights"]) <= set(current_features.scenario_type)


def test_short_window_and_no_injected_episodes(tmp_path):
    config = yaml.safe_load((ROOT / "config/generator.yaml").read_text())
    config["n_customers"] = 20
    config["days"] = 10
    config["behaviour"]["p_late_joiner"] = 0
    config["legit_episodes"] = dict(emergency=0, new_phone=0, travel=0)
    old = tmp_path / "old"
    new = tmp_path / "new"
    ArchivedSimulator(copy.deepcopy(config)).run(str(old))
    generate_dataset(copy.deepcopy(config), new)
    for filename in ["events_log.csv", "vouchers_truth.csv", "customers.csv", "customers_latent.csv"]:
        assert (old / filename).read_bytes() == (new / filename).read_bytes()
