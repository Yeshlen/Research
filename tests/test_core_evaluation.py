import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core_evaluation import metrics, choose_threshold, split_stage, band_cutoffs, risk_bands, scenario_rates


def test_metrics_known_confusion_counts_and_no_alerts():
    result = metrics([1,1,0,0], [.9,.3,.8,.1], .5)
    assert (result['tp'],result['fp'],result['fn'],result['tn']) == (1,1,1,1)
    assert result['precision'] == result['recall'] == result['f1'] == result['fpr'] == .5
    none = metrics([1,0], [.1,.2], .9)
    assert none['precision'] == none['recall'] == none['f1'] == 0


def test_threshold_matches_exhaustive_validation_search():
    rng = np.random.default_rng(42)
    y = np.array([0]*40 + [1]*10)
    scores = np.round(rng.random(50), 1)
    threshold, result = choose_threshold(y, scores)
    expected = max((metrics(y, scores, t)['f1'], t) for t in np.unique(scores))
    assert result['f1'] == pytest.approx(expected[0])
    assert threshold == expected[1]
    with pytest.raises(ValueError): choose_threshold([0,0], [.2,.4])


def test_redemption_time_does_not_enter_earlier_period():
    start = pd.Timestamp('2025-07-01')
    df = pd.DataFrame({'voucher_id':[1,2,3,4], 'redeemed':[1,1,1,0],
        'created_at':[start+pd.Timedelta(days=x) for x in (89,119,120,10)],
        'redeemed_at':[start+pd.Timedelta(days=90), start+pd.Timedelta(days=120), start+pd.Timedelta(days=180), pd.NaT]})
    a = split_stage(df,'A',start)
    b = split_stage(df,'B',start)
    assert set(a[0].voucher_id) == {1,4}
    assert list(b[0].voucher_id) == []
    assert list(b[1].voucher_id) == [1]
    assert list(b[2].voucher_id) == [2]
    assert all(len(set(x.voucher_id) & set(y.voucher_id)) == 0 for x,y in [(b[0],b[1]),(b[1],b[2])])


def test_bands_are_exactly_consistent_with_alert_threshold():
    y = np.array([0,1,0,1,0,1])
    scores = np.array([.1,.3,.4,.6,.8,.9])
    threshold, _ = choose_threshold(y,scores)
    cuts = band_cutoffs(y,scores,threshold)
    assert cuts['Medium'] <= cuts['High'] <= cuts['Critical']
    values = np.array([0,cuts['Medium'],threshold,cuts['Critical'],1])
    assert np.array_equal(np.isin(risk_bands(values,cuts),['High','Critical']), values >= threshold)


def test_scenario_counts_preserve_different_group_sizes():
    part = pd.DataFrame({'scenario_type':['normal','normal','scam_split'], 'is_scam':[0,0,1]}, index=[10,20,30])
    rates = scenario_rates(part,{'Hybrid':np.array([True,False,True])},'A').set_index('scenario_type')
    assert rates.loc['normal','n'] == 2
    assert rates.loc['normal','rate'] == .5
    assert rates.loc['scam_split','interpretation'] == 'Recall'


def test_fitted_bundle_rejects_changed_input_but_ignores_notebook_outputs(tmp_path):
    import json
    from core_evaluation import fingerprint, check_fitted_bundle
    for name in ['data/synthetic/cardless_features.csv', 'config/generator.yaml', 'src/models.py', 'src/core_evaluation.py']:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    notebook = tmp_path/'notebooks/02_fit_models.ipynb'
    notebook.parent.mkdir()
    cell = {'cell_type': 'code', 'source': ['fit()'], 'outputs': []}
    notebook.write_text(json.dumps({'cells': [cell]}))
    bundle = {'fingerprint': fingerprint(tmp_path)}
    cell['outputs'] = [{'text': 'Finished'}]
    notebook.write_text(json.dumps({'cells': [cell]}))
    check_fitted_bundle(bundle, tmp_path)
    (tmp_path/'data/synthetic/cardless_features.csv').write_text('different data')
    with pytest.raises(ValueError, match='Run notebook 02'):
        check_fitted_bundle(bundle, tmp_path)
