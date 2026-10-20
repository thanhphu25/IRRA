import ast
import json
from pathlib import Path


def test_kaggle_notebook_python_cells():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / 'notebooks/irra_lgsp_icfg_kaggle.ipynb').read_text(encoding='utf-8'))
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and not cell['outputs']
            ast.parse(''.join(cell['source']), filename=f'cell-{i}')
