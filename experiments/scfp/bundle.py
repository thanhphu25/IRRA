"""Package the current V2 source tree for Kaggle without committing/pushing."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def bundle(destination):
    root = Path(__file__).resolve().parents[2]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    paths = []
    for folder in ('model', 'datasets', 'processor', 'solver', 'utils', 'experiments/scfp', 'tests', 'notebooks'):
        paths += [p for p in (root / folder).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts
                  and p.suffix in ('.py', '.md', '.txt', '.yaml', '.ipynb', '.json')]
    paths += [root / p for p in ('train.py', 'test.py', 'README.md', 'LICENSE', 'pytest.ini',
                                 'THIRD_PARTY_NOTICES.md', 'data/bpe_simple_vocab_16e6.txt.gz')]
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if path.is_file():
                # ZIP entries and their manifest must agree on Kaggle/Linux even
                # when this bundle is produced from a Windows workspace.
                name = path.relative_to(root).as_posix()
                archive.write(path, name)
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        archive.writestr('source_manifest.json', json.dumps(hashes, indent=2))
    print(f'Source bundle: {destination} ({len(hashes)} files)')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/tmp/IRRA_scfp_v2_source.zip')
    bundle(parser.parse_args().output)
