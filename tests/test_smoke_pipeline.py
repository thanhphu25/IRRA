import json

from PIL import Image

from experiments.lgsp_tbpr.smoke_test import run_smoke


def test_file_dataset_transforms_tokenizer_smoke(make_model, tmp_path):
    # Synthetic JPEG/annotation files through the real ICFG loader and tokenizer.
    images = tmp_path / 'data/ICFG-PEDES/imgs'
    images.mkdir(parents=True)
    records = []
    for identity in range(4):
        for instance in range(2):
            name = f'{identity}_{instance}.jpg'
            Image.new('RGB', (24, 48), color=(30 + identity * 40, 80, 150)).save(images / name)
            records.append(dict(id=identity, file_path=name,
                captions=['A person wearing a blue jacket and black pants.'],
                split='train' if identity < 3 else 'test'))
    (images.parent / 'ICFG-PEDES.json').write_text(json.dumps(records), encoding='utf-8')
    template = make_model('frequency', vocab_size=49408, root_dir=str(tmp_path / 'data'),
                          dataset_name='ICFG-PEDES', val_dataset='holdout', batch_size=2,
                          num_workers=0, output_dir=str(tmp_path / 'smoke'))
    args = template.args
    del template
    result = run_smoke(args)
    assert result['mode'] == 'frequency'
    assert result['input_shape'] == [2, 3, 32, 16]
    assert result['image_feature_shape'] == result['text_feature_shape'] == [2, 64]
    assert set(result['losses']) == {'sdm_loss', 'id_loss', 'mlm_loss'}
    assert result['prompt_gradients']['frequency.logits'] > 0
    assert (tmp_path / 'smoke/smoke_metrics.json').is_file()
