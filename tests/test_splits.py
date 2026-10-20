from datasets.splits import train_validation_holdout


def test_holdout_identity_disjoint_reproducible():
    data = [(pid, pid * 2 + img, f'{pid}_{img}.jpg', cap)
            for pid in range(10) for img in range(2) for cap in ('first', 'second')]
    train, val, held = train_validation_holdout(data, 0.2, 42)
    assert (train, val, held) == train_validation_holdout(data, 0.2, 42)
    assert not {r[0] for r in train}.intersection(held)
    assert len(train) == 32 and len(val['img_paths']) == 4 and len(val['captions']) == 8
    assert len(set(val['img_paths'])) == 4
