"""Download the frozen release bundle and verify each checkpoint before use."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def install(bundle, root, manifest):
    with zipfile.ZipFile(bundle) as z:
        for item in manifest['files']:
            relative = Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'weights':
                raise ValueError('Invalid manifest path')
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_suffix(target.suffix + '.partial')
            with z.open(relative.as_posix()) as src, staged.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if staged.stat().st_size != item['bytes'] or digest(staged) != item['sha256']:
                staged.unlink()
                raise ValueError(f'Checkpoint checksum failed: {relative}')
            staged.replace(target)
            print(f'Verified {relative}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, help='Use an already downloaded release ZIP')
    args = p.parse_args()
    manifest = json.loads((ROOT / 'weights/manifest.json').read_text(encoding='utf8'))
    bundle_info = json.loads((ROOT / 'weights/bundle.json').read_text(encoding='utf8'))
    with tempfile.TemporaryDirectory(prefix='nkai-weights-') as tmp:
        archive = args.archive or Path(tmp) / manifest['asset']
        if not args.archive:
            print('Downloading official repository release bundle...')
            with urlopen(bundle_info['url'], timeout=120) as response, archive.open('wb') as out:
                shutil.copyfileobj(response, out)
        if digest(archive) != bundle_info['sha256']:
            raise ValueError('Release archive SHA-256 mismatch')
        install(archive, ROOT, manifest)


if __name__ == '__main__':
    main()
