"""Package existing research data for a GitHub Release; never regenerate it."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

FILES = ['cardless_features.csv', 'customers.csv', 'customers_latent.csv',
         'events_log.csv', 'dataset_metadata.json', 'vouchers_truth.csv']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', action='store_true', help='Package saved models and the app results table instead of raw data.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / ('report/core' if args.runtime else 'data/synthetic')
    paths = {'fitted_models.joblib': 'models/fitted_models.joblib',
             'model_comparison.csv': 'tables/model_comparison.csv'} if args.runtime else {name: name for name in FILES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, relative in paths.items():
                info = archive.gettarinfo(str(data / relative), arcname=name)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with (data / relative).open('rb') as stream:
                    archive.addfile(info, stream)
    manifest = {
        'url': 'https://github.com/Yeshlen/Research/releases/download/demo-data-v1/' + args.output.name,
        'sha256': sha256(args.output),
        'files': {name: {'sha256': sha256(data / relative), 'bytes': (data / relative).stat().st_size, **({'destination': relative} if args.runtime else {})} for name, relative in paths.items()},
    }
    (root / 'deployment' / ('runtime_release.json' if args.runtime else 'data_release.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Created {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)')
