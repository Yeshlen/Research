"""A clean hosted checkout must get verified data without modifying local research files."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'app'))
from deployment_data import ensure_dataset
from session_examples import remember_example, session_examples


def release(tmp_path, member_name='customers.csv'):
    payload = b'customer_id\n1\n'
    archive = tmp_path / 'dataset.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    root = tmp_path / 'checkout'
    (root / 'deployment').mkdir(parents=True)
    manifest = {'url': archive.as_uri(), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'files': {'customers.csv': {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}}}
    (root / 'deployment/data_release.json').write_text(json.dumps(manifest))
    return root, archive, manifest, payload


def test_clean_checkout_downloads_and_reuses_without_network(tmp_path, monkeypatch):
    root, _, _, payload = release(tmp_path)
    ensure_dataset(root)
    assert (root / 'data/synthetic/customers.csv').read_bytes() == payload
    monkeypatch.setattr('deployment_data.urlopen', lambda *a, **k: pytest.fail('Unexpected download'))
    ensure_dataset(root)


def test_bad_download_installs_nothing(tmp_path):
    root, archive, _, _ = release(tmp_path)
    archive.write_bytes(b'corrupted archive')
    with pytest.raises(ValueError, match='checksum'):
        ensure_dataset(root)
    assert not (root / 'data/synthetic').exists()


def test_archive_cannot_write_outside_dataset(tmp_path):
    root, _, _, _ = release(tmp_path, '../escaped.csv')
    with pytest.raises(ValueError, match='unexpected'):
        ensure_dataset(root)
    assert not (root / 'data/synthetic').exists()
    assert not (tmp_path / 'escaped.csv').exists()


def test_partial_local_dataset_is_not_overwritten(tmp_path):
    root, _, manifest, _ = release(tmp_path)
    folder = root / 'data/synthetic'
    folder.mkdir(parents=True)
    (folder / 'customers.csv').write_text('different data')
    manifest['files']['missing.csv'] = {'sha256': 'unused', 'bytes': 0}
    (root / 'deployment/data_release.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='differs'):
        ensure_dataset(root)
    assert (folder / 'customers.csv').read_text() == 'different data'


def test_generated_examples_belong_only_to_one_visitor():
    first, second = {}, {}
    row = pd.DataFrame({'voucher_id': [10], 'amount': [100]})
    remember_example(first, row)
    row.loc[0, 'amount'] = 999
    assert session_examples(second).empty
    assert session_examples(first).iloc[0].amount == 100
    with pytest.raises(ValueError, match='already been saved'):
        remember_example(first, pd.DataFrame({'voucher_id': [10]}))
