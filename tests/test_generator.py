"""Generator sanity tests on a small simulated population. Run with: pytest -q"""
import os
import sys

import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import extract  # noqa: E402
from simulator import generate_dataset  # noqa: E402
from validate import constraint_violations  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    with open(os.path.join(ROOT, "config", "generator.yaml")) as f:
        c = yaml.safe_load(f)
    c["n_customers"] = 800
    return c


@pytest.fixture(scope="module")
def dataset(cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("sim")
    generate_dataset(cfg, out)
    return out, extract(str(out), cfg)


def test_no_constraint_violations(cfg, dataset):
    _, df = dataset
    bad = {k: v for k, v in constraint_violations(df, cfg).items() if v}
    assert not bad, bad


def test_same_seed_same_data(cfg, dataset, tmp_path):
    out, _ = dataset
    generate_dataset(cfg, tmp_path)
    a = pd.read_csv(out / "events_log.csv")
    b = pd.read_csv(tmp_path / "events_log.csv")
    pd.testing.assert_frame_equal(a, b)


def test_every_voucher_has_truth_and_features(dataset):
    out, df = dataset
    truth = pd.read_csv(out / "vouchers_truth.csv")
    assert len(df) == len(truth)
    assert df["is_scam"].notna().all()


def test_all_scam_types_present(cfg, dataset):
    _, df = dataset
    assert set(cfg["scam"]["type_weights"]) <= set(df.scenario_type)


def test_scam_prevalence_in_expected_range(dataset):
    _, df = dataset
    assert 0.005 < df.is_scam.mean() < 0.03


def test_features_use_only_the_past(dataset):
    """The first voucher of every customer must look like it has no history."""
    _, df = dataset
    first = df.sort_values("created_at").groupby("customer_id").head(1)
    assert (first.history_count == 0).all()
    assert (first.recipient_known == 0).all()
    assert (first.amount_ratio_to_avg_90d == 1.0).all()


def test_scams_only_after_warmup(cfg, dataset):
    _, df = dataset
    assert (df[df.is_scam == 1].day >= cfg["warmup_days"]).all()
