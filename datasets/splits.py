"""Identity-disjoint train-only validation; preserve original classifier indices."""
import numpy as np


def train_validation_holdout(train, fraction=0.1, seed=42):
    identities = sorted({row[0] for row in train})
    if not 0 < fraction < 1 or len(identities) < 2:
        raise ValueError('Holdout needs 0 < fraction < 1 and at least two identities')
    count = min(len(identities) - 1, max(1, round(len(identities) * fraction)))
    selected = set(np.random.RandomState(seed).choice(identities, count, replace=False).tolist())
    remaining = [row for row in train if row[0] not in selected]
    held = [row for row in train if row[0] in selected]
    images = {}
    for pid, image_id, path, _ in held:
        images[image_id] = (pid, path)
    validation = dict(image_pids=[p for p, _ in images.values()],
                      img_paths=[p for _, p in images.values()],
                      caption_pids=[row[0] for row in held], captions=[row[3] for row in held])
    return remaining, validation, sorted(selected)
