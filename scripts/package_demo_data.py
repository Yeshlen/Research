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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / 'data/synthetic'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name in FILES:
                info = archive.gettarinfo(str(data / name), arcname=name)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                with (data / name).open('rb') as stream:
                    archive.addfile(info, stream)
    manifest = {
        'url': 'https://github.com/Yeshlen/Research/releases/download/demo-data-v1/' + args.output.name,
        'sha256': sha256(args.output),
        'files': {name: {'sha256': sha256(data / name), 'bytes': (data / name).stat().st_size} for name in FILES},
    }
    (root / 'deployment/data_release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Created {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)')
