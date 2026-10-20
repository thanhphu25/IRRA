import json
import pytest
from experiments.lgsp_tbpr import prepare_kaggle as prepare


def test_discovery_and_expected_notebook_layout(tmp_path, monkeypatch):
    source = tmp_path / 'input/persons'
    images = source / 'imgs/train'
    images.mkdir(parents=True)
    (images / 'one.jpg').write_bytes(b'path-existence-fixture')
    annotation = source / 'ICFG_PEDES.json'
    annotation.write_text(json.dumps([dict(id=0, split='train', file_path='train/one.jpg',
        captions=['A person'])]), encoding='utf-8')
    linked = []
    monkeypatch.setattr(prepare, 'link', lambda a, b, directory=False: linked.append((a, b, directory)))
    info = prepare.prepare_icfg(tmp_path / 'input', tmp_path / 'working/data')
    assert info['splits'] == {'train': 1}
    assert linked[0][0] == annotation.resolve()
    assert linked[0][1].name == 'ICFG-PEDES.json'
    assert linked[1][0] == images.parent.resolve() and linked[1][2]


def test_discovery_refuses_ambiguous_checkpoints(tmp_path):
    for name in ('a', 'b'):
        path = tmp_path / name / 'best.pth'
        path.parent.mkdir()
        path.write_bytes(b'path-existence-fixture')
    with pytest.raises(ValueError, match='Expected one'):
        prepare.find_checkpoint(tmp_path)
    assert prepare.find_checkpoint(tmp_path, tmp_path / 'a/best.pth') == (tmp_path / 'a/best.pth').resolve()
