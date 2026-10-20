"""Discover mounted ICFG-PEDES and prepare IRRA's symlink layout safely."""
import json
from pathlib import Path


def unique_path(candidates, label):
    candidates = sorted({Path(p).resolve() for p in candidates})
    if len(candidates) != 1:
        raise ValueError(f'Expected one {label}; set its explicit config path. Found: {candidates}')
    return candidates[0]


def link(source, target, directory=False):
    if target.is_symlink():
        if target.resolve() == source.resolve():
            return
        raise FileExistsError(f'{target} points elsewhere; choose a new DATA_ROOT')
    if target.exists():
        raise FileExistsError(f'Will not overwrite existing {target}; choose a new DATA_ROOT')
    target.symlink_to(source, target_is_directory=directory)


def prepare_icfg(input_root='/kaggle/input', data_root='/kaggle/working/data',
                 annotation_source=None, images_source=None):
    root = Path(input_root)
    annotation = Path(annotation_source).resolve() if annotation_source else unique_path(
        list(root.rglob('ICFG-PEDES.json')) + list(root.rglob('ICFG_PEDES.json')), 'annotation')
    records = json.loads(annotation.read_text(encoding='utf-8'))
    if not records:
        raise ValueError('Empty annotation file')
    samples = [r['file_path'] for r in records[:1000]]
    if images_source:
        images = Path(images_source).resolve()
    else:
        candidates = [annotation.parent / 'imgs', annotation.parent / 'images', annotation.parent]
        candidates += [p for p in root.rglob('imgs') if p.is_dir()]
        images = unique_path([p for p in candidates if p.is_dir() and all((p / s).is_file() for s in samples)], 'image root')
    missing = [s for s in samples if not (images / s).is_file()]
    if missing:
        raise FileNotFoundError(f'Images missing under {images}: {missing[:5]}')
    destination = Path(data_root) / 'ICFG-PEDES'
    destination.mkdir(parents=True, exist_ok=True)
    link(annotation, destination / 'ICFG-PEDES.json')
    link(images, destination / 'imgs', directory=True)
    splits = {s: sum(r['split'] == s for r in records) for s in sorted({r['split'] for r in records})}
    result = dict(annotation=str(annotation), images=str(images), destination=str(destination), splits=splits)
    print(json.dumps(result, indent=2))
    return result


def find_checkpoint(input_root='/kaggle/input', explicit=None):
    path = Path(explicit).resolve() if explicit else unique_path(Path(input_root).rglob('best.pth'), 'baseline best.pth')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_root', default='/kaggle/input')
    parser.add_argument('--data_root', default='/kaggle/working/data')
    parser.add_argument('--annotation_source')
    parser.add_argument('--images_source')
    args = parser.parse_args()
    prepare_icfg(**vars(args))


if __name__ == '__main__':
    main()
