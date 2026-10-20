from pathlib import Path
import ast
import json
import zipfile

import nbformat

from experiments.scfp.bundle import bundle


def test_notebook_schema_python_cells_branch_and_safe_defaults():
    notebook = nbformat.read('notebooks/irra_scfp_v2_icfg_kaggle.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    config = {}
    exec(notebook.cells[1].source, config)
    assert config['BRANCH'] == 'feat/scfp-v2'
    assert config['EPOCHS'] == 1 and config['EXPERIMENT'] == 'v2'
    assert not config['RUN_FINAL_TEST'] and not config['RUN_BASELINE_EVALUATION']
    assert config['SOURCE_DIR'] is None


def test_notebook_finds_kaggle_extracted_source(tmp_path):
    import zipfile
    archive = bundle(tmp_path / 'source.zip')
    inputs = tmp_path / 'input'
    extracted = inputs / 'dataset'
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(extracted)
    notebook = nbformat.read('notebooks/irra_scfp_v2_icfg_kaggle.ipynb', as_version=4)
    setup = notebook.cells[2].source
    start = setup.index('root = Path(INPUT_ROOT)')
    stop = setup.index('REPO = Path(')
    env = dict(Path=Path, INPUT_ROOT=str(inputs), SOURCE_ZIP=None, SOURCE_DIR=None)
    exec(setup[start:stop], env)
    assert env['SOURCE_DIR'] == str(extracted)
    assert env['SOURCE_ZIP'] is None


def test_source_bundle_complete(tmp_path):
    path = bundle(tmp_path / 'IRRA_scfp_v2_source.zip')
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('source_manifest.json'))
        assert {'model/scfp/feature_frequency.py', 'data/bpe_simple_vocab_16e6.txt.gz',
                'tests/fixtures/irra_build_main.py', 'train.py'} <= manifest.keys()
        assert not any(n.endswith('.pth') or '/.git/' in n for n in archive.namelist())
