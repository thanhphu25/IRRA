"""Kaggle-ready SCFP V2 audit. No training, optimizer, or checkpoint writes.

Run --help for real checkpoint usage; --synthetic explicitly exercises plumbing
with a random tiny model and cannot produce evidence about a trained checkpoint.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
from unittest.mock import patch

import torch
import yaml

from experiments.scfp.audit_metrics import (fingerprint, frequency_measurement, gradient_measurement,
                                            intervention_measurement, slot_measurement, stats, verify_bypass)
from experiments.scfp.audit_visuals import frequency_plot, slot_plots
from experiments.scfp.diagnostics import serializable, write_json


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _tiny_clip(*_args, **_kwargs):
    from model.clip_model import CLIP
    config = dict(embed_dim=512, image_resolution=(32, 16), vision_layers=3, vision_width=64,
                  vision_patch_size=8, stride_size=8, context_length=8, vocab_size=128,
                  transformer_width=64, transformer_heads=1, transformer_layers=1)
    return CLIP(**config), config


def load_model(cli):
    from utils.options import get_args
    from utils.checkpoint import Checkpointer, strip_prefix_if_present, ARCHITECTURE_FIELDS
    build = importlib.import_module('model.build')
    args = get_args([])
    if cli.synthetic:
        args.scfp_mode, args.scfp_frequency_layer = 'v2', 1
        args.img_size, args.stride_size, args.vocab_size, args.text_length = (32, 16), 8, 128, 8
        args.cmt_depth, args.precision = 1, 'fp32'
        args.device = cli.device
        with patch.object(build, 'build_CLIP_from_openai_pretrained', _tiny_clip):
            model = build.build_model(args, 4)
        return model.to(cli.device).eval(), dict(status='synthetic_random_model', config=vars(args))
    if not cli.checkpoint:
        raise ValueError('--checkpoint is required unless --synthetic is explicit')
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint.get('config', {})
    external = yaml.safe_load(Path(cli.config_file).read_text(encoding='utf-8')) if cli.config_file else {}
    if not saved and not external:
        raise ValueError('Checkpoint has no embedded config: supply its exact --config-file')
    for key in (*ARCHITECTURE_FIELDS, 'temperature', 'scfp_slot_loss_weight', 'scfp_diversity_weight',
                'freeze_backbone', 'precision'):
        a, b = saved.get(key), external.get(key)
        if key in saved and key in external and serializable(a) != serializable(b):
            raise ValueError(f'Embedded and external config differ for {key}: {a} != {b}')
    for key, value in (external | saved).items():
        setattr(args, key, value)
    if args.scfp_mode != 'v2':
        raise ValueError(f'Audit requires the trained V2 model, got {args.scfp_mode}')
    state = strip_prefix_if_present(checkpoint.get('model', checkpoint))
    classes = state['classifier.weight'].shape[0] if 'classifier.weight' in state else checkpoint.get('num_classes', 1)
    metadata = {k: checkpoint[k] for k in ('epoch', 'scfp_mode', 'num_classes', 'best_rank1') if k in checkpoint}
    del state, checkpoint
    args.device, args.distributed, args.training = cli.device, False, False
    if cli.device == 'cpu' or cli.precision == 'fp32':
        args.precision = 'fp32'
    args.backbone_checkpoint = str(Path(cli.checkpoint).resolve())
    # Constructor only reconstructs shapes. Checkpointer strictly validates and
    # loads every tensor; it never restores an optimizer, scheduler or RNG state.
    model = build.build_model(args, classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(cli.checkpoint)
    model.to(cli.device).eval()
    if model.base_model.visual.scfp_frequency.bands != 4 or model.scfp_slots.shared_queries.shape[0] != 4:
        raise ValueError('This four-band/four-slot audit expects the supplied V2 configuration')
    return model, dict(status='loaded_strictly', checkpoint=str(Path(cli.checkpoint).resolve()),
                       checkpoint_sha256=file_hash(cli.checkpoint), metadata=metadata,
                       compatibility=checkpointer.last_report, config=vars(args),
                       saved_precision=saved.get('precision', external.get('precision')),
                       audit_precision=args.precision)


def sample_records(annotations, num_identities, images_per_identity, seed):
    """Fixed random identities/images; one random caption per unique image."""
    groups = defaultdict(list)
    for index, record in enumerate(annotations):
        if record['split'] == 'train' and record['captions']:
            groups[int(record['id'])].append((index, record))
    rng = random.Random(seed)
    identities = rng.sample(sorted(groups), min(num_identities, len(groups)))
    selected = []
    for pid in identities:
        unique = {record['file_path']: (index, record) for index, record in groups[pid]}
        candidates = sorted(unique.values(), key=lambda x: x[0])
        for index, record in rng.sample(candidates, min(images_per_identity, len(candidates))):
            caption_index = rng.randrange(len(record['captions']))
            selected.append(dict(annotation_index=index, pid=pid, file_path=record['file_path'],
                                 caption_index=caption_index, caption=record['captions'][caption_index]))
    rng.shuffle(selected)
    if len({r['pid'] for r in selected}) < 2:
        raise ValueError('At least two training identities are needed for meaningful negative pairs')
    return selected


def representative_indices(records, count):
    """Preselect caption-length quantiles, prefer different identities; no scores."""
    ordered = sorted(range(len(records)), key=lambda i: (len(records[i]['caption'].split()), i))
    chosen, pids = [], set()
    count = min(count, len(records))
    for j in range(count):
        target = round(j * (len(ordered) - 1) / max(1, count - 1))
        available = [i for i in ordered if i not in chosen and records[i]['pid'] not in pids]
        if not available:
            available = [i for i in ordered if i not in chosen]
        index = min(available, key=lambda i: abs(ordered.index(i) - target))
        chosen.append(index)
        pids.add(records[index]['pid'])
    return chosen


def gradient_indices(records, batch_size, count, seed):
    rng = random.Random(seed + 1)
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record['pid']].append(index)
    pids = sorted(groups)
    rng.shuffle(pids)
    for indices in groups.values():
        rng.shuffle(indices)
    batches = []
    for number in range(count):
        chosen = []
        # Include at least two identities and repeated positives where available.
        for j in range(min(len(pids), max(2, batch_size // 2))):
            pid = pids[(number * max(2, batch_size // 2) + j) % len(pids)]
            chosen.extend(groups[pid][:2])
        remaining = [i for i in range(len(records)) if i not in chosen]
        rng.shuffle(remaining)
        chosen = (chosen + remaining)[:batch_size]
        rng.shuffle(chosen)
        batches.append(chosen)
    return batches


def prepare_samples(cli, args):
    if cli.synthetic:
        records = [dict(annotation_index=i, pid=i // 4, file_path=f'synthetic_{i}', caption_index=0,
                        caption='synthetic ' + ' '.join(f'token{j}' for j in range(2 + i % 4))) for i in range(16)]
        images = torch.randn(16, 3, *args.img_size)
        ids = torch.zeros(16, args.text_length, dtype=torch.long)
        for i in range(16):
            length = 2 + i % 4
            ids[i, 0], ids[i, length + 1] = args.vocab_size - 2, args.vocab_size - 1
            ids[i, 1:length + 1] = torch.randint(1, 100, (length,))
        return records, images, ids, None, 'synthetic'
    if not cli.root_dir:
        raise ValueError('--root-dir must contain ICFG-PEDES/ICFG-PEDES.json and ICFG-PEDES/imgs/')
    from datasets.build import build_transforms
    from datasets.bases import tokenize
    from utils.simple_tokenizer import SimpleTokenizer
    from PIL import Image
    root = Path(cli.root_dir) / 'ICFG-PEDES'
    annotations = json.loads((root / 'ICFG-PEDES.json').read_text(encoding='utf-8'))
    records = sample_records(annotations, cli.num_identities, cli.images_per_identity, cli.seed)
    transform = build_transforms(args.img_size, is_train=False)
    images = []
    tokenizer = SimpleTokenizer()
    ids = []
    for record in records:
        with Image.open(root / 'imgs' / record['file_path']) as image:
            images.append(transform(image.convert('RGB')))
        ids.append(tokenize(record['caption'], tokenizer, args.text_length))
    return records, torch.stack(images), torch.stack(ids), tokenizer, file_hash(root / 'ICFG-PEDES.json')


def collect(model, images, ids, pids, batch_size, device):
    frequency, relevance, feature_chunks, frequency_chunks, text_global = model.base_model.visual.scfp_frequency, [], [], [], []
    def frequency_hook(module, arguments, output):
        frequency_chunks.append(frequency_measurement(module, arguments[0], arguments[1], output))
    def relevance_hook(_module, _arguments, output):
        relevance.append(output.detach().squeeze(-1).cpu())
    hooks = [frequency.register_forward_hook(frequency_hook),
             model.scfp_slots.relevance_head.register_forward_hook(relevance_hook)]
    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                captions = ids[start:start + batch_size].to(device)
                visual, text = model.base_model(images[start:start + batch_size].to(device), captions)
                f = model.scfp_slots(visual[:, 1:], text, captions)
                if not all(bool(torch.isfinite(v).all()) for v in f.values()):
                    raise FloatingPointError('Nonfinite slot output')
                feature_chunks.append({key: value.cpu() for key, value in f.items()})
                text_global.append(text[torch.arange(len(text), device=device), captions.argmax(-1)].float().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    features = {key: torch.cat([f[key] for f in feature_chunks]) for key in feature_chunks[0]}
    per_image = ('logits', 'weights', 'band_gains', 'response', 'input_energy', 'input_energy_fraction',
                 'output_energy', 'residual_energy_fraction', 'residual_ratio', 'band_rms_maps',
                 'input_rms_map', 'residual_rms_map')
    frequency_data = dict(frequency_chunks[0])
    frequency_data.update({key: torch.cat([f[key] for f in frequency_chunks]) for key in per_image})
    for key in ('reconstruction_max_error', 'imaginary_max', 'module_output_max_error'):
        frequency_data[key] = max(f[key] for f in frequency_chunks)
    for key in ('coverage_exact', 'conjugate_symmetric', 'cls_unchanged'):
        frequency_data[key] = all(f[key] for f in frequency_chunks)
    slots = slot_measurement(features, pids, torch.cat(relevance), model.logit_scale, ids, model.args.vocab_size)
    return frequency_data, features, slots, torch.cat(text_global)


def frequency_report(data):
    omit = {'band_rms_maps', 'input_rms_map', 'residual_rms_map'}
    result = {key: value for key, value in data.items() if key not in omit}
    weights, gains = data['weights'], data['band_gains']
    result.update(status='measured',
                  formula='G_b=1+tanh(alpha_raw)*tanh(R*softmax(logits/temperature)_b-1)',
                  bands=['low: radius [0,.25)', 'lower-mid: [.25,.5)',
                         'upper-mid: [.5,.75)', 'high: [.75,1]'],
                  radius='sqrt((fy/.5)^2+(fx/.5)^2)/sqrt(2), cycles per patch-grid step',
                  normalization='Disjoint binary partition, sum_b masks=1; ortho FFT. No area normalization.',
                  weight_mean=weights.mean(0), weight_std_across_images=weights.std(0, unbiased=False),
                  weight_range_across_images=weights.amax(0) - weights.amin(0),
                  gain_mean=gains.mean(0), gain_std_across_images=gains.std(0, unbiased=False),
                  gain_range_across_images=gains.amax(0) - gains.amin(0),
                  entropy=stats(-(weights * weights.clamp_min(1e-30).log()).sum(-1)),
                  entropy_max=__import__('math').log(weights.shape[1]),
                  sample_pairs_mean_absolute_gain_difference=stats(torch.pdist(gains, p=1) / gains.shape[1]),
                  zero_gain_bins=int((data['response'] == 0).sum()),
                  empty_bands=int((data['mask_bin_counts'] == 0).sum()),
                  note='Low softmax weight is not zero frequency gain. Small inter-image ranges imply weak adaptivity on this subset only.')
    return result


def placeholder_plot(path, reason):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_axis_off()
    ax.text(0.02, 0.5, 'UNAVAILABLE\n' + reason[:400], wrap=True, transform=ax.transAxes)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_report(output, reports, provenance):
    freq, slots, gradients, intervention = (reports[k] for k in ('frequency', 'slot', 'gradient', 'intervention'))
    status = '\n'.join(f'- {name}: `{item["status"]}`' for name, item in reports.items())
    text = f'''# SCFP V2 diagnostic audit — không huấn luyện

Nguồn kết quả: **{provenance['evidence']}**. Tập con train cố định, đã được checkpoint thấy trong huấn luyện; không phải validation độc lập hay official test. Seed {provenance['seed']}; {provenance['sample_count']} cặp; manifest và hash nằm trong mọi JSON.

{status}

Đã kiểm tra mã nguồn: bốn hard radial masks chia toàn bộ FFT bins; gain thực tế là `1+tanh(alpha_raw)*tanh(4w−1)`. Softmax lệch không có nghĩa xóa băng tần. Relevance có floor 0.1, sigmoid và chuẩn hóa tổng; khi điểm các slot tương tự nhau, trọng số ít ảnh hưởng tới local score. Loss SDM cũng có thể gần bão hòa trên batch dễ. Cần đọc số đo đạo hàm trong gradient_analysis.json để phân biệt hai giả thuyết này với sigmoid bão hòa.

Lỗi biên xác nhận: text mask loại ID 0 như padding dù BPE thật có ID 0 là `!` và có thể xuất hiện trước EOT. Số vị trí bị loại trên tập con nằm trong slot_analysis.json/token_mask_audit. Chưa chứng minh lỗi này gây giảm Rank-1; không thay đổi mask trong audit.

Giới hạn xác nhận từ mã nguồn: diversity phạt cosine embeddings, không trực tiếp buộc attention khác nhau. Query dùng chung và local loss cùng index chưa chứng minh slot cùng ngữ nghĩa. Slots chỉ hỗ trợ loss; encode_image/encode_text dùng CLS/EOT cho retrieval. Slot loss không đi trực tiếp qua final CLS/EOT slice, nhưng vẫn có thể đi qua các tham số backbone dùng chung ở trước.

Gate nhận trung bình không gian, không trực tiếp nhận năng lượng từng băng; perturbation tần số có trung bình 0 không đổi descriptor. Penalty diversity có gradient 0 tại collapse hoàn toàn. Đây là giới hạn thiết kế, chưa phải chẩn đoán nguyên nhân checkpoint hiện tại.

ON/OFF dùng `forward(tokens, grid_size)` và cờ enabled của module thật, chỉ trong eval/no_grad, khôi phục cờ bằng finally. Đây là can thiệp suy luận trên cùng trọng số, không thay thế ablation huấn luyện riêng. Không dùng số test để chọn siêu tham số.

Chưa thể kết luận nguyên nhân giảm Rank-1 từ một checkpoint/batch hoặc attention map. V2.1 nên ưu tiên kiểm tra adaptivity/gain, mức bão hòa loss và gradient backbone; sau đó mới thử từng thay đổi có đối chứng: attention diversity, cạnh tranh slot hoặc relevance đơn giản hơn. Không triển khai thay đổi kiến trúc trong audit này.

Tham số/buffers không đổi: `{provenance.get('state_unchanged')}`. Checkpoint trên đĩa không đổi: `{provenance.get('checkpoint_file_unchanged')}`. Không tạo optimizer, không optimizer.step, không lưu trọng số.

Lệnh đã chạy: `{provenance['command']}`. Chi tiết môi trường, commit, precision, load compatibility và lỗi/unavailable nằm trong JSON.
'''
    if freq['status'] == 'measured':
        text += f'\nGain trung bình: {serializable(freq["gain_mean"])}; alpha hiệu dụng: {float(freq["effective_alpha"]):.8f}.\n'
    if slots['status'] == 'measured':
        text += f'\nCosine slot off-diagonal (visual/text): {slots["visual_embedding_off_diagonal_cosine"]["mean"]:.5f} / {slots["text_embedding_off_diagonal_cosine"]["mean"]:.5f}.\n'
    if intervention['status'] == 'measured':
        r = intervention['retrieval']
        text += f'\nTập con train: R1 ON={r["rank1_on_percent"]:.4f}%, OFF={r["rank1_off_percent"]:.4f}%, ON−OFF={r["rank1_on_minus_off_percentage_points"]:.4f} điểm phần trăm.\n'
    (output / 'SCFP_V2_AUDIT.md').write_text(text, encoding='utf-8')


def run(cli):
    import sys
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cli.seed)
    random.seed(cli.seed)
    if cli.device == 'cpu':
        torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    contract = verify_bypass(cli.device)
    model, model_info = load_model(cli)
    records, images, ids, tokenizer, annotation_hash = prepare_samples(cli, model.args)
    pids = torch.tensor([r['pid'] for r in records])
    selected = representative_indices(records, cli.examples)
    grad_indices = gradient_indices(records, cli.gradient_batch_size, cli.gradient_batches, cli.seed)
    manifest = dict(split='training_subset_seen_during_training', seed=cli.seed, records=records,
                    visualization_indices=selected, gradient_batch_indices=grad_indices,
                    selection='Random identities/images/caption; visualization chosen before inference by caption-length quantiles and identity coverage',
                    annotation_sha256=annotation_hash)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(output / 'subset_manifest.json', manifest)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    provenance = dict(evidence='SYNTHETIC RANDOM MODEL / INPUTS — không phải checkpoint đã train' if cli.synthetic else 'TRAINED CHECKPOINT',
                      command=' '.join(sys.argv), seed=cli.seed, sample_count=len(records),
                      manifest_sha256=manifest_hash, manifest=manifest, model=model_info,
                      git_commit=commit.stdout.strip(), torch_version=torch.__version__,
                      device=cli.device, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(cli.device) if cli.device.startswith('cuda') else None,
                      python=sys.version, gradient_mode='eval with autograd; deterministic resize/normalize, no training augmentation')
    source_root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve(), Path(__file__).with_name('audit_metrics.py'),
                    Path(__file__).with_name('audit_visuals.py'), source_root / 'model/build.py',
                    source_root / 'model/clip_model.py', source_root / 'model/objectives.py',
                    source_root / 'utils/simple_tokenizer.py', *sorted((source_root / 'model/scfp').glob('*.py'))]
    provenance['source_sha256'] = {str(p.relative_to(source_root)): file_hash(p) for p in source_paths}
    before = fingerprint(model)
    provenance['state_sha256_before'] = before
    frequency = model.base_model.visual.scfp_frequency
    initial_flags = frequency.enabled, frequency.capture
    frequency.enabled, frequency.capture = True, False
    reports = {}
    def measure(name, operation):
        print(f'Audit: {name}', flush=True)
        try:
            return operation()
        except Exception as error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reports[name] = dict(status='unavailable', reason=f'{type(error).__name__}: {error}')
            print(f'{name}: {reports[name]["reason"]}', flush=True)
            return None
    try:
        collected = measure('collection', lambda: collect(model, images, ids, pids, cli.batch_size, cli.device))
        if collected is not None:
            frequency_data, features, slots, text_embeddings = collected
            reports['frequency'], reports['slot'] = frequency_report(frequency_data), dict(status='measured', **slots)
            labels = [f'subset {i} | PID {r["pid"]} | anno {r["annotation_index"]}' for i, r in enumerate(records)]
            captions = [r['caption'] for r in records]
            measure('frequency_plot', lambda: frequency_plot(output / 'frequency_response.png', frequency_data,
                                                              images, selected, labels, cli.synthetic))
            measure('slot_plots', lambda: slot_plots(output / 'visual_slot_attention.png', output / 'text_slot_attention.png',
                                                     features, images, ids, selected, labels, captions,
                                                     frequency_data['grid'], tokenizer, cli.synthetic))
            def intervene():
                image_batches = (images[start:start + cli.batch_size].to(cli.device)
                                 for start in range(0, len(images), cli.batch_size))
                return intervention_measurement(model, image_batches, text_embeddings, pids)
            intervention = measure('intervention', intervene)
            if intervention is not None:
                reports['intervention'] = intervention
        else:
            for name in ('frequency', 'slot', 'intervention'):
                reports[name] = dict(status='unavailable', reason=reports['collection']['reason'])
        grad_reports = []
        for number, indices in enumerate(grad_indices):
            batch = dict(images=images[indices].to(cli.device), caption_ids=ids[indices].to(cli.device),
                         pids=pids[indices].to(cli.device))
            result = measure(f'gradient_batch_{number}', lambda: gradient_measurement(model, batch))
            grad_reports.append(dict(indices=indices, result=result or reports[f'gradient_batch_{number}']))
        measured = sum(r['result']['status'] == 'measured' for r in grad_reports)
        reports['gradient'] = dict(status='measured' if measured == len(grad_reports) and measured else 'partial' if measured else 'unavailable',
                                   batches=grad_reports, measured_batches=measured,
                                   note='Norms/cosines reported separately per batch, not averaged into a fictitious dataset gradient.')
    finally:
        frequency.enabled, frequency.capture = initial_flags
        frequency.last_features, frequency.last_diagnostics = {}, {}
        model.scfp_features = {}
        provenance['state_sha256_after'] = fingerprint(model)
        provenance['state_unchanged'] = provenance['state_sha256_after'] == before
        provenance['parameter_grad_buffers_untouched'] = all(p.grad is None for p in model.parameters())
        provenance['checkpoint_file_unchanged'] = (file_hash(cli.checkpoint) == model_info['checkpoint_sha256']) if cli.checkpoint else None
    reports['intervention']['synthetic_bypass_verification'] = contract
    for name, filename in (('frequency', 'frequency_analysis.json'), ('slot', 'slot_analysis.json'),
                           ('gradient', 'gradient_analysis.json'), ('intervention', 'frequency_intervention.json')):
        write_json(output / filename, dict(provenance=provenance, **reports[name]))
    for filename in ('frequency_response.png', 'visual_slot_attention.png', 'text_slot_attention.png'):
        if not (output / filename).exists():
            placeholder_plot(output / filename, str({k: v for k, v in reports.items() if v['status'] == 'unavailable'}))
    write_report(output, reports, provenance)
    assert provenance['state_unchanged'] and provenance['parameter_grad_buffers_untouched']
    assert provenance['checkpoint_file_unchanged'] is not False
    print(f'Audit complete: {output}', flush=True)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Existing trusted SCFP V2 best.pth, read-only')
    parser.add_argument('--config-file', help='Exact run configs.yaml, optional if embedded in checkpoint')
    parser.add_argument('--root-dir', help='Parent of ICFG-PEDES/ (prepared by prepare_kaggle)')
    parser.add_argument('--output-dir', required=True, help='New directory; existing output is never overwritten')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['checkpoint', 'fp32'], default='checkpoint')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num-identities', type=int, default=32)
    parser.add_argument('--images-per-identity', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradient-batch-size', type=int, default=8)
    parser.add_argument('--gradient-batches', type=int, default=3)
    parser.add_argument('--examples', type=int, default=6)
    parser.add_argument('--synthetic', action='store_true', help='Explicitly random tiny model/inputs; no trained-model claims')
    cli = parser.parse_args(argv)
    if min(cli.num_identities, cli.images_per_identity, cli.batch_size, cli.gradient_batch_size,
           cli.gradient_batches, cli.examples) < 1 or cli.gradient_batch_size < 4 or cli.num_identities < 2:
        parser.error('Positive counts required; at least 2 identities and gradient batch size >=4')
    if cli.synthetic and (cli.checkpoint or cli.config_file):
        parser.error('--synthetic cannot be combined with a real checkpoint/config')
    reports = run(cli)
    return 0 if all(r['status'] == 'measured' for r in reports.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
