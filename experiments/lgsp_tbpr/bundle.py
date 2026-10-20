"""Package the current working tree for Kaggle without pushing Git changes."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def bundle(output):
    root = Path(__file__).resolve().parents[2]
    files = []
    for directory in ('model', 'solver', 'utils', 'datasets', 'processor', 'experiments', 'tests', 'notebooks'):
        files.extend(p for p in (root / directory).rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix in ('.py', '.md', '.sh', '.txt', '.ipynb'))
    for name in ('train.py', 'test.py', 'visualize.py', 'README.md', 'LICENSE',
                 'THIRD_PARTY_NOTICES.md', 'pytest.ini', 'run_irra.sh', 'data/bpe_simple_vocab_16e6.txt.gz'):
        files.append(root / name)
    manifest = {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip(),
                'files': {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in manifest['files']:
            z.write(root / name, 'IRRA/' + name)
        z.writestr('IRRA/bundle_manifest.json', json.dumps(manifest, indent=2))
    print(f'{output.resolve()} ({output.stat().st_size / 1024**2:.2f} MiB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='dist/irra_lgsp_source.zip')
    bundle(parser.parse_args().output)
