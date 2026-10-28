"""Download the published synthetic dataset when a hosted checkout has no local data."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from tempfile import TemporaryDirectory
from urllib.request import urlopen


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ensure_release(root, manifest_name, destination):
    root = Path(root)
    manifest = json.loads((root / 'deployment' / manifest_name).read_text())
    folder = root / destination
    files = manifest['files']
    targets = {}
    for name, details in files.items():
        relative = PurePosixPath(details.get('destination', name))
        if Path(name).name != name or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid release filename.')
        targets[name] = folder.joinpath(*relative.parts)
    if len(set(targets.values())) != len(targets):
        raise ValueError('Duplicate release destinations.')
    # Local notebook users keep their existing files and need no network connection.
    if all(path.is_file() for path in targets.values()):
        return
    for name, details in files.items():
        path = targets[name]
        if path.exists() and file_hash(path) != details['sha256']:
            raise ValueError('Existing dataset differs from the release. Use a separate clean checkout.')
    with TemporaryDirectory(prefix='cardless-data-') as temporary:
        temporary = Path(temporary)
        archive_path = temporary / 'dataset.tar.gz'
        with urlopen(manifest['url'], timeout=120) as response, archive_path.open('wb') as output:
            shutil.copyfileobj(response, output)
        if file_hash(archive_path) != manifest['sha256']:
            raise ValueError('Dataset download failed its checksum. Please retry.')
        unpacked = temporary / 'unpacked'
        unpacked.mkdir()
        with tarfile.open(archive_path, 'r:gz') as archive:
            members = archive.getmembers()
            if len(members) != len(files) or {m.name for m in members} != set(files):
                raise ValueError('Dataset archive contains unexpected files.')
            for member in members:
                expected = files[member.name]
                if not member.isfile() or member.size != expected['bytes']:
                    raise ValueError('Invalid dataset archive entry.')
                path = unpacked / member.name
                with archive.extractfile(member) as source, path.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                if file_hash(path) != expected['sha256']:
                    raise ValueError('An extracted dataset file failed its checksum.')
        # Validate everything before installing anything. Never overwrite local research files.
        folder.mkdir(parents=True, exist_ok=True)
        for name in files:
            target = targets[name]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(unpacked / name, target)


def ensure_dataset(root):
    _ensure_release(root, 'data_release.json', 'data/synthetic')


def ensure_runtime(root):
    _ensure_release(root, 'runtime_release.json', 'report/core')
