"""Scientific/contract checks for diagnostics, without training the model."""
import importlib
import json

import pytest
import torch

from experiments.scfp.audit import (gradient_indices, load_model, main, representative_indices,
                                    sample_records, collect)
from experiments.scfp.audit_metrics import (fingerprint, frequency_measurement, frequency_mode,
                                            gradient_measurement, verify_bypass, attention_metrics)
from model.scfp.feature_frequency import AdaptiveFeatureFrequency, radial_masks
from model.scfp.local_alignment import local_similarity
from model.scfp.diversity import slot_diversity
from tests.helpers import tiny_args, tiny_clip, synthetic_batch


@pytest.fixture
def audit_model(monkeypatch):
    build = importlib.import_module('model.build')
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    torch.manual_seed(12)
    return build.build_model(tiny_args(), 3).eval()


def test_bypass_real_signature_dtype_cls_patches_and_exception_restore():
    assert len(verify_bypass()['cases']) == 6
    m = AdaptiveFeatureFrequency(8).eval()
    with pytest.raises(RuntimeError, match='no_grad'):
        with frequency_mode(m, False):
            pass
    with torch.no_grad():
        with pytest.raises(ValueError, match='sentinel'):
            with frequency_mode(m, False):
                raise ValueError('sentinel')
        assert m.enabled
        m.enabled = False
        with frequency_mode(m, True):
            assert m.enabled
        assert not m.enabled


@pytest.mark.parametrize('shape', [(24, 8), (9, 5)])
def test_frequency_actual_transfer_parseval_and_masks(shape):
    m = AdaptiveFeatureFrequency(8).eval()
    with torch.no_grad():
        m.alpha_raw.fill_(0.03155)
        m.gate.net[-1].weight.zero_()
        m.gate.net[-1].bias.copy_(torch.tensor([0.0434, 0.9179, 0.0157, 0.0230]).log())
        x = torch.randn(3, 1 + shape[0] * shape[1], 8)
        y = m(x, shape)
        d = frequency_measurement(m, x, shape, y)
    assert d['coverage_exact'] and d['conjugate_symmetric'] and d['cls_unchanged']
    assert d['module_output_max_error'] < 2e-6
    assert d['reconstruction_max_error'] < 2e-6 and d['imaginary_max'] < 2e-6
    assert int(d['mask_bin_counts'].sum()) == shape[0] * shape[1]
    torch.testing.assert_close(d['band_gains'][0], torch.tensor([.9785992, 1.0312394, .9768521, .9772861]))
    # Orthonormal FFT gives the same residual energy ratio in either domain.
    torch.testing.assert_close(d['residual_ratio'].square(), d['residual_energy_fraction'], atol=1e-7, rtol=1e-4)
    assert (d['band_gains'] > .97).all()  # low softmax does not remove bands


def test_spatial_mean_gate_cannot_see_zero_mean_frequency_perturbation():
    torch.manual_seed(1)
    m = AdaptiveFeatureFrequency(8)
    grid = torch.randn(2, 8, 24, 8)
    checkerboard = ((torch.arange(24)[:, None] + torch.arange(8)[None]) % 2) * 2. - 1
    changed = grid + 3 * checkerboard
    torch.testing.assert_close(m.gate(grid)[0], m.gate(changed)[0], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(torch.fft.fft2(grid).abs(), torch.fft.fft2(changed).abs())


def test_identical_slot_scores_make_relevance_unidentifiable():
    visual = torch.nn.functional.normalize(torch.randn(4, 1, 32), dim=-1).expand(-1, 4, -1)
    text = torch.nn.functional.normalize(torch.randn(4, 1, 32), dim=-1).expand(-1, 4, -1)
    logits = torch.randn(4, 4, requires_grad=True)
    raw = .1 + .9 * logits.sigmoid()
    relevance = raw / raw.sum(-1, keepdim=True)
    score = local_similarity(visual, text, relevance)
    torch.testing.assert_close(score, local_similarity(visual, text, torch.ones_like(relevance)))
    gradient = torch.autograd.grad(score.square().sum(), logits)[0]
    assert gradient.abs().max() < 1e-7


def test_diversity_is_stationary_at_exact_collapse():
    x = torch.zeros(2, 4, 8)
    x[:, :, 0] = 1
    x.requires_grad_()
    loss = slot_diversity(x)
    assert loss == 1
    assert not torch.autograd.grad(loss, x)[0].any()


def test_token_zero_is_real_bpe_content_and_production_mask_drops_it():
    from datasets.bases import tokenize
    from utils.simple_tokenizer import SimpleTokenizer
    from model.scfp.semantic_slots import text_content_mask
    tokenizer = SimpleTokenizer()
    ids = tokenize('person !$', tokenizer, 12)[None]
    assert tokenizer.encode('!$') == [0, 259]
    zero_content = ids.eq(0) & (torch.arange(12)[None] < ids.argmax(-1)[:, None])
    assert zero_content.any()
    assert not text_content_mask(ids)[zero_content].any()


def test_gradient_audit_matches_production_losses_and_reaches_shared_parameters(audit_model):
    batch = synthetic_batch()
    state = fingerprint(audit_model)
    with torch.no_grad():
        expected = audit_model(batch)
    report = gradient_measurement(audit_model, batch)
    for name in ('slot', 'sdm', 'diversity'):
        assert report['losses'][name] == pytest.approx(float(expected[name + '_loss']), rel=1e-6, abs=1e-6)
    for modality in ('visual_backbone', 'text_backbone'):
        group = report['parameter_groups'][modality]
        assert group['slot_norm'] > 0 and group['sdm_norm'] > 0
        assert group['both_nonzero_tensors'] > 10 and group['all_finite']
        assert -1 <= group['cosine'] <= 1
    node = report['representation_gradients']['slot']
    assert node['visual_sequence_detail']['cls_norm'] == 0
    assert node['text_sequence_detail']['eot_norm'] == 0
    assert node['visual_sequence_detail']['patch_norm'] > 0
    assert node['text_sequence_detail']['content_norm'] > 0
    assert node['relevance_logits'] > 0  # true output node, not a detached/sibling squeeze
    assert fingerprint(audit_model) == state
    assert all(p.grad is None for p in audit_model.parameters())


def test_collection_chunking_and_attention_mask(audit_model):
    b = synthetic_batch()
    before = fingerprint(audit_model)
    f, features, slots, embeddings = collect(audit_model, b['images'], b['caption_ids'], b['pids'], 2, 'cpu')
    assert f['weights'].shape == (4, 4) and f['module_output_max_error'] < 1e-6
    assert slots['all_equal_pid_pairs_positive'] and slots['target_row_sum_max_error'] == 0
    assert slots['text_attention']['masked_attention_mass'] == 0
    assert embeddings.shape == (4, 512)
    assert fingerprint(audit_model) == before
    # An empty description cannot spuriously count as distinct attention.
    empty = attention_metrics(torch.zeros(2, 4, 8), torch.zeros(2, 8, dtype=torch.bool))
    assert empty['empty_descriptions'] == 2 and empty['off_diagonal_cosine'] is None


def test_sampling_fixed_before_scores_and_training_only():
    annotations = [dict(id=i // 5, split='train', file_path=f'{i}.jpg', captions=['a person', 'longer caption person walks'])
                   for i in range(40)]
    annotations.append(dict(id=100, split='test', file_path='test.jpg', captions=['never select']))
    samples = sample_records(annotations, 6, 3, 2026)
    assert samples == sample_records(annotations, 6, 3, 2026)
    assert len(samples) == 18 and all(r['pid'] != 100 for r in samples)
    assert len({r['file_path'] for r in samples}) == 18
    representatives = representative_indices(samples, 6)
    assert len({samples[i]['pid'] for i in representatives}) == 6
    for indices in gradient_indices(samples, 8, 3, 2026):
        assert len(set(indices)) == 8
        assert len({samples[i]['pid'] for i in indices}) >= 2


def test_checkpoint_loading_embedded_config_strict_and_readonly(audit_model, tmp_path):
    from argparse import Namespace
    checkpoint = tmp_path / 'trained.pth'
    torch.save(dict(model=audit_model.state_dict(), config=vars(audit_model.args), scfp_mode='v2', num_classes=3), checkpoint)
    original = checkpoint.read_bytes()
    model, info = load_model(Namespace(synthetic=False, checkpoint=str(checkpoint), config_file=None,
                                     device='cpu', precision='checkpoint'))
    assert info['status'] == 'loaded_strictly' and not info['compatibility']['missing_keys']
    assert fingerprint(model) == fingerprint(audit_model)
    assert checkpoint.read_bytes() == original


def test_synthetic_cli_delivers_outputs_and_refuses_overwrite(tmp_path):
    output = tmp_path / 'audit'
    command = ['--synthetic', '--device', 'cpu', '--output-dir', str(output), '--examples', '2', '--gradient-batches', '1']
    assert main(command) == 0
    for name in ('frequency_analysis', 'slot_analysis', 'gradient_analysis', 'frequency_intervention'):
        data = json.loads((output / (name + '.json')).read_text())
        assert data['status'] == 'measured'
        assert data['provenance']['state_unchanged'] and data['provenance']['parameter_grad_buffers_untouched']
        assert data['provenance']['evidence'].startswith('SYNTHETIC')
    for name in ('frequency_response.png', 'visual_slot_attention.png', 'text_slot_attention.png', 'SCFP_V2_AUDIT.md'):
        assert (output / name).stat().st_size > 100
    with pytest.raises(FileExistsError):
        main(command)


def test_file_checkpoint_dataset_cli_and_unavailable_reporting(monkeypatch, tmp_path):
    """Exercise real file/tokenizer path with a random checkpoint fixture, no data download."""
    from PIL import Image
    from model.clip_model import CLIP
    build = importlib.import_module('model.build')
    def full_vocab_clip(*args, **kwargs):
        config = dict(embed_dim=64, image_resolution=(32, 16), vision_layers=3, vision_width=64,
                      vision_patch_size=8, stride_size=8, context_length=8, vocab_size=49408,
                      transformer_width=64, transformer_heads=1, transformer_layers=1)
        return CLIP(**config), config
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', full_vocab_clip)
    args = tiny_args()
    args.vocab_size, args.loss_names = 49408, 'sdm'
    model = build.build_model(args, 3).eval()
    checkpoint = tmp_path / 'random_fixture.pth'
    torch.save(dict(model=model.state_dict(), config=vars(args), scfp_mode='v2', num_classes=3), checkpoint)
    before = checkpoint.read_bytes()
    dataset = tmp_path / 'ICFG-PEDES'
    (dataset / 'imgs').mkdir(parents=True)
    records = []
    for i in range(6):
        Image.new('RGB', (16, 32), (i * 30, 80, 100)).save(dataset / 'imgs' / f'{i}.png')
        records.append(dict(id=i // 2, split='train', file_path=f'{i}.png', captions=['person !$']))
    (dataset / 'ICFG-PEDES.json').write_text(json.dumps(records))
    audit = importlib.import_module('experiments.scfp.audit')
    def unavailable(*_args, **_kwargs):
        raise RuntimeError('deliberate unavailable fixture')
    monkeypatch.setattr(audit, 'gradient_measurement', unavailable)
    output = tmp_path / 'real_file_path_audit'
    assert main(['--checkpoint', str(checkpoint), '--root-dir', str(tmp_path), '--output-dir', str(output),
                 '--device', 'cpu', '--num-identities', '3', '--images-per-identity', '2',
                 '--examples', '2', '--gradient-batches', '1']) == 2
    assert checkpoint.read_bytes() == before
    gradient = json.loads((output / 'gradient_analysis.json').read_text())
    assert gradient['status'] == 'unavailable' and gradient['measured_batches'] == 0
    assert 'deliberate unavailable' in gradient['batches'][0]['result']['reason']
    slots = json.loads((output / 'slot_analysis.json').read_text())
    assert slots['status'] == 'measured'
    assert slots['token_mask_audit']['content_id_zero_count'] == 6
    assert slots['token_mask_audit']['excluded_non_special_content_count'] == 6
    intervention = json.loads((output / 'frequency_intervention.json').read_text())
    assert intervention['status'] == 'measured' and intervention['on_repeat_exact']
    assert intervention['provenance']['checkpoint_file_unchanged']


def test_audit_notebook_schema_defaults_and_no_training():
    import ast
    import nbformat
    notebook = nbformat.read('notebooks/irra_scfp_v2_audit_kaggle.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    assert len(code) == 3
    for source in code:
        ast.parse(source)
    config = {}
    exec(code[0], config)
    assert config['BRANCH'] == 'feat/scfp-v2'
    assert config['GRADIENT_BATCHES'] == 3 and config['EXAMPLES'] == 6
    assert config['PRECISION'] == 'checkpoint'
    assert 'experiments.scfp.audit' in code[2]
    assert 'kaggle_train' not in ''.join(code) and 'torch.save' not in ''.join(code)
