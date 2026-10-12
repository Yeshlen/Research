"""Scoring sanity tests on a small simulated population. Run with: pytest -q"""
import numpy as np
import pytest

from test_generator import cfg, dataset  # noqa: F401  (shared fixtures)
from models import STAGE_A_FEATURES, STAGE_B_FEATURES, band, hybrid_score, isolation_forest, rule_score


@pytest.fixture(scope="module")
def df(dataset):  # noqa: F811
    return dataset[1]


def test_bands():
    assert [band(s) for s in (0, 39, 40, 69, 70, 89, 90, 100)] == \
        ["Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]


@pytest.mark.parametrize("stage", ["A", "B"])
def test_rule_score_range_and_reasons(df, stage):
    score, reasons = rule_score(df, stage)
    assert score.min() >= 0 and score.max() <= 100
    assert all((s > 0) == bool(r) for s, r in zip(score, reasons))
    assert score[df.is_scam.to_numpy() == 1].mean() > score[df.is_scam.to_numpy() == 0].mean()


def test_stage_b_adds_redemption_rules(df):
    assert (rule_score(df, "B")[0] >= rule_score(df, "A")[0]).all()


def test_isolation_forest_percentile(df):
    train = df[df.split == "train"]
    for feats, part in ((STAGE_A_FEATURES, df), (STAGE_B_FEATURES, df[df.redeemed == 1])):
        s = isolation_forest(feats).fit(train).score(part)
        assert s.min() >= 0 and s.max() <= 1
    h = hybrid_score(np.array([0, 100]), np.array([1.0, 0.0]), 0.5)
    assert np.allclose(h, [0.5, 0.5])
