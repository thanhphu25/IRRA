"""Read-only SCFP V2 measurements, using the production forward/loss definitions."""
from contextlib import contextmanager
import hashlib

import torch
import torch.nn.functional as F

from model import objectives
from model.scfp.feature_frequency import AdaptiveFeatureFrequency, radial_masks
from model.scfp.local_alignment import local_alignment_loss, local_similarity, positive_targets
from model.scfp.diversity import slot_diversity


def stats(x):
    x = x.detach().double().cpu()
    if not x.numel():
        return None
    if not torch.isfinite(x).all():
        return dict(status='unavailable', reason='nonfinite values')
    return dict(mean=x.mean().item(), std=x.std(unbiased=False).item(),
                min=x.min().item(), max=x.max().item())


def fingerprint(model):
    """Hash parameters AND buffers without retaining a second model in memory."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def frequency_mode(module, enabled):
    """Inference-only bypass for forward(tokens, grid_size); no module replacement."""
    if not isinstance(module, AdaptiveFeatureFrequency):
        raise TypeError('Only the inspected AdaptiveFeatureFrequency implementation is supported')
    if module.training or torch.is_grad_enabled():
        raise RuntimeError('Frequency intervention requires eval() and no_grad()/inference_mode()')
    previous = module.enabled
    module.enabled = enabled
    try:
        yield
    finally:
        module.enabled = previous


def verify_bypass(device='cpu'):
    """Exercise the REAL two-argument forward on even/odd rectangular token grids."""
    cases = []
    with torch.inference_mode():
        for grid in ((24, 8), (9, 5)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                module = AdaptiveFeatureFrequency(8).to(device).eval()
                tokens = torch.randn(2, 1 + grid[0] * grid[1], 8, device=device).to(dtype)
                with frequency_mode(module, False):
                    result = module(tokens, grid)
                    assert result is tokens
                    assert torch.equal(result[:, :1], tokens[:, :1])
                    assert torch.equal(result[:, 1:], tokens[:, 1:])
                    assert result.dtype == dtype and result.shape == tokens.shape
                assert module.enabled
                cases.append(dict(grid=list(grid), dtype=str(dtype), exact_identity=True))
    return dict(status='measured_synthetic_contract', cases=cases)


@torch.no_grad()
def frequency_measurement(module, tokens, grid_size, output=None):
    """Reconstruct actual transfer function and verify it against the module output."""
    h, w = grid_size
    grid = tokens[:, 1:].float().transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
    logits, weights = module.gate(grid)
    if not torch.isfinite(weights).all() or not torch.isfinite(grid).all():
        raise FloatingPointError('Nonfinite frequency input/gate response')
    masks = radial_masks(h, w, module.bands, grid.device)
    gains = 1 + module.alpha * (module.bands * weights - 1).tanh()
    response = torch.einsum('br,rhw->bhw', gains, masks)
    spectrum = torch.fft.fft2(grid, norm='ortho')
    filtered = torch.fft.ifft2(spectrum * response[:, None], norm='ortho')
    band_energy = torch.einsum('bchw,rhw->br', spectrum.abs().square(), masks)
    total_energy = band_energy.sum(-1, keepdim=True).clamp_min(1e-30)
    restored = torch.fft.ifft2(spectrum, norm='ortho')
    residual = filtered.real - grid
    # Compact per-patch RMS for plots, not a single selected feature channel.
    band_maps = torch.stack([
        torch.fft.ifft2(spectrum * m, norm='ortho').real.square().mean(1).sqrt()
        for m in masks], dim=1)
    conjugate = masks.index_select(1, (-torch.arange(h, device=grid.device)) % h)
    conjugate = conjugate.index_select(2, (-torch.arange(w, device=grid.device)) % w)
    expected = torch.cat((tokens[:, :1], filtered.real.flatten(2).transpose(1, 2).to(tokens.dtype)), 1)
    return dict(
        grid=list(grid_size), alpha_raw=module.alpha_raw.detach().cpu(),
        effective_alpha=module.alpha.detach().cpu(), logits=logits.cpu(), weights=weights.cpu(),
        band_gains=gains.cpu(), response=response.cpu(), masks=masks.cpu(),
        mask_bin_counts=masks.sum((1, 2)).cpu(), coverage_exact=bool((masks.sum(0) == 1).all()),
        conjugate_symmetric=bool(torch.equal(masks, conjugate)),
        dc_band=int(masks[:, 0, 0].argmax()),
        reconstruction_max_error=float((restored.real - grid).abs().max()),
        imaginary_max=float(filtered.imag.abs().max()),
        module_output_max_error=None if output is None else float((output - expected).abs().max()),
        cls_unchanged=None if output is None else bool(torch.equal(output[:, :1], tokens[:, :1])),
        input_energy=band_energy.cpu(), input_energy_fraction=(band_energy / total_energy).cpu(),
        output_energy=(band_energy * gains.square()).cpu(),
        residual_energy_fraction=((band_energy * (gains - 1).square()).sum(-1) /
                                  total_energy[:, 0]).cpu(),
        residual_ratio=(residual.flatten(1).norm(dim=1) / grid.flatten(1).norm(dim=1).clamp_min(1e-30)).cpu(),
        band_rms_maps=band_maps.cpu(),
        input_rms_map=grid.square().mean(1).sqrt().cpu(),
        residual_rms_map=residual.square().mean(1).sqrt().cpu())


def off_diagonal(matrix):
    k = matrix.shape[-1]
    return matrix[..., ~torch.eye(k, dtype=torch.bool, device=matrix.device)]


def attention_metrics(attention, valid=None):
    attention = attention.float()
    if valid is None:
        valid = torch.ones(attention.shape[0], attention.shape[-1], dtype=torch.bool,
                           device=attention.device)
    entropy = -(attention * attention.clamp_min(1e-30).log()).sum(-1)
    normalizer = valid.sum(-1).clamp_min(2).float().log()[:, None]
    nonempty = valid.any(-1)
    unit = F.normalize(attention, dim=-1)
    cosine = unit @ unit.transpose(-1, -2)
    p, q = attention[:, :, None], attention[:, None, :]
    midpoint = (p + q) * 0.5
    js = 0.5 * ((p * (p.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1) +
                (q * (q.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1))
    # Top-patch/token collisions supplement head-averaged cosine and JS.
    peaks = attention.argmax(-1)
    collisions = peaks[:, :, None].eq(peaks[:, None, :]).float()
    return dict(entropy=stats(entropy[nonempty]), normalized_entropy=stats((entropy / normalizer)[nonempty]),
                off_diagonal_cosine=stats(off_diagonal(cosine)[nonempty]),
                off_diagonal_js_divergence=stats(off_diagonal(js)[nonempty]),
                argmax_collision_fraction=stats(off_diagonal(collisions)[nonempty]),
                masked_attention_mass=float(attention.masked_select(~valid[:, None].expand_as(attention)).sum()),
                empty_descriptions=int((~nonempty).sum()),
                row_sum_max_error=float((attention.sum(-1) - nonempty[:, None].float()).abs().max()))


@torch.no_grad()
def slot_measurement(features, pids, relevance_logits, logit_scale, token_ids=None, vocab_size=49408):
    visual, text = features['visual_slots'], features['text_slots']
    rel, raw = features['slot_relevance'], features['slot_relevance_raw']
    matching = torch.einsum('bkd,bld->bkl', visual, text)
    same = matching.diagonal(dim1=-2, dim2=-1)
    pair_scores = torch.einsum('ikd,jkd->ijk', F.normalize(visual.float(), dim=-1),
                               F.normalize(text.float(), dim=-1))
    similarity = local_similarity(visual, text, rel)
    uniform = pair_scores.mean(-1)
    positive = pids[:, None].eq(pids[None])
    probs_i = (similarity * logit_scale).softmax(-1)
    probs_t = (similarity.t() * logit_scale).softmax(-1)
    sig = relevance_logits.sigmoid()
    targets = positive_targets(pids)
    result = dict(
        logit_scale=float(logit_scale),
        visual_embedding_off_diagonal_cosine=stats(off_diagonal(visual @ visual.transpose(-1, -2))),
        text_embedding_off_diagonal_cosine=stats(off_diagonal(text @ text.transpose(-1, -2))),
        visual_attention=attention_metrics(features['visual_attention']),
        text_attention=attention_metrics(features['text_attention'], features['text_content_mask']),
        paired_matching_matrix_mean=matching.mean(0).cpu(),
        paired_matching_matrix_per_sample=matching.cpu(),
        same_index_cosine=stats(same), cross_index_cosine=stats(off_diagonal(matching)),
        same_index_argmax_fraction=float(matching.argmax(-1).eq(torch.arange(visual.shape[1], device=visual.device)).float().mean()),
        visual_diversity_loss_unweighted=float(slot_diversity(visual)),
        text_diversity_loss_unweighted=float(slot_diversity(text)),
        relevance_weights=rel.cpu(), relevance_raw=raw.cpu(), relevance_logits=relevance_logits.cpu(),
        relevance_entropy=stats(-(rel * rel.clamp_min(1e-30).log()).sum(-1)),
        relevance_std_across_slots=stats(rel.std(-1, unbiased=False)),
        sigmoid_derivative=stats(sig * (1 - sig)),
        sigmoid_saturation_fraction_threshold_001=float(((sig < 0.01) | (sig > 0.99)).float().mean()),
        per_pair_slot_score_std=stats(pair_scores.std(-1, unbiased=False)),
        learned_minus_uniform_local_score=stats((similarity - uniform).abs()),
        positive_local_similarity=stats(similarity[positive]),
        negative_local_similarity=stats(similarity[~positive]),
        local_margin=None if not (~positive).any() else float(similarity[positive].mean() - similarity[~positive].mean()),
        sdm_negative_probability_mass_i2t=stats((probs_i * ~positive).sum(-1)),
        sdm_negative_probability_mass_t2i=stats((probs_t * ~positive).sum(-1)),
        identity_counts=positive.sum(-1).cpu(), positive_target_rows=targets.cpu(),
        target_row_sum_max_error=float((targets.sum(-1) - 1).abs().max()),
        all_equal_pid_pairs_positive=bool((targets.gt(0) == positive).all()),
        negative_pair_count=int((~positive).sum()),
        note='Attention is averaged over heads; correspondence scores do not establish named body-part semantics.')
    if token_ids is not None:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None]
        end = token_ids.eq(vocab_size - 1).int().argmax(-1)
        boundary_content = ((positions > 0) & (positions < end[:, None]) &
                            token_ids.ne(vocab_size - 2) & token_ids.ne(vocab_size - 3))
        dropped = boundary_content & ~features['text_content_mask']
        result['token_mask_audit'] = dict(
            content_id_zero_count=int((boundary_content & token_ids.eq(0)).sum()),
            excluded_non_special_content_count=int(dropped.sum()),
            excluded_content_positions=dropped.nonzero().cpu(),
            eot_count_per_caption=token_ids.eq(vocab_size - 1).sum(-1).cpu(),
            note='Real CLIP BPE id 0 is ! and can occur before EOT (e.g. !$ -> [0,259]). '
                 'Production ids.ne(0) drops it as padding. Counts are empirical; no mask is changed.')
    return result


def _group(name):
    if name.startswith('base_model.visual.scfp_frequency.'):
        return 'frequency'
    if name.startswith('base_model.visual.'):
        return 'visual_backbone'
    if name.startswith('base_model.'):
        return 'text_backbone'
    if name.startswith('scfp_slots.relevance_head.'):
        return 'relevance_head'
    return 'slot_heads'


def gradient_comparison(left, right):
    used_left = [g for g in left if g is not None]
    used_right = [g for g in right if g is not None]
    l2 = sum(float(g.double().square().sum()) for g in used_left)
    r2 = sum(float(g.double().square().sum()) for g in used_right)
    dot = sum(float(a.double().flatten() @ b.double().flatten())
              for a, b in zip(left, right) if a is not None and b is not None)
    return dict(slot_norm=l2 ** 0.5, sdm_norm=r2 ** 0.5,
                slot_to_sdm_norm_ratio=(l2 / r2) ** 0.5 if r2 else None,
                cosine=dot / (l2 * r2) ** 0.5 if l2 and r2 else None,
                slot_connected_tensors=len(used_left), sdm_connected_tensors=len(used_right),
                both_nonzero_tensors=sum(a is not None and b is not None and
                                         bool(a.any()) and bool(b.any()) for a, b in zip(left, right)),
                all_finite=all(bool(torch.isfinite(g).all()) for g in used_left + used_right))


def gradient_measurement(model, batch):
    """autograd.grad only; no optimizer, backward(), .grad accumulation or updates.

    Reuse production encoders, slot module, local/SDM/diversity losses and weights.
    Do not run unrelated MLM/ID heads. Final CLS/EOT direct derivatives can be zero
    for slots while earlier shared backbone parameters still receive gradients.
    """
    if model.training:
        raise RuntimeError('Audit uses eval mode, with gradients explicitly enabled')
    named = [(n, p) for n, p in model.named_parameters()
             if p.requires_grad and n.startswith(('base_model.', 'scfp_slots.'))]
    if not named:
        return dict(status='unavailable', reason='No trainable shared/head parameters')
    captured = {}
    def save_logits(_module, _args, output):
        captured['logits'] = output
    handle = model.scfp_slots.relevance_head.register_forward_hook(save_logits)
    try:
        with torch.enable_grad():
            images, texts = model.base_model(batch['images'], batch['caption_ids'])
            features = model.scfp_slots(images[:, 1:], texts, batch['caption_ids'])
            image_global = images[:, 0].float()
            eot = batch['caption_ids'].argmax(-1)
            text_global = texts[torch.arange(len(texts), device=texts.device), eot].float()
            similarity = local_similarity(features['visual_slots'], features['text_slots'], features['slot_relevance'])
            losses = dict(slot=local_alignment_loss(similarity, batch['pids'], model.logit_scale) * model.args.scfp_slot_loss_weight,
                          sdm=objectives.compute_sdm(image_global, text_global, batch['pids'], model.logit_scale),
                          diversity=(slot_diversity(features['visual_slots']) + slot_diversity(features['text_slots'])) *
                          0.5 * model.args.scfp_diversity_weight)
            if not all(bool(torch.isfinite(loss)) for loss in losses.values()):
                raise FloatingPointError('Nonfinite diagnostic loss')
            node_names = ['visual_sequence', 'text_sequence', 'local_scores', 'relevance_weights',
                          'relevance_raw', 'relevance_logits']
            nodes = [images, texts, similarity, features['slot_relevance'],
                     features['slot_relevance_raw'], captured['logits']]
            # Nodes lacking requires_grad (e.g. a deliberately frozen backbone) are
            # reported unavailable, not silently unfrozen for this diagnostic.
            active_nodes = [(n, x) for n, x in zip(node_names, nodes) if x.requires_grad]
            all_targets = [p for _, p in named] + [x for _, x in active_nodes]
            grads, node_report = {}, {}
            for loss_name, loss in losses.items():
                values = torch.autograd.grad(loss, all_targets, retain_graph=True, allow_unused=True)
                if not all(g is None or bool(torch.isfinite(g).all()) for g in values):
                    raise FloatingPointError(f'Nonfinite {loss_name} gradient')
                grads[loss_name] = [None if g is None else g.detach().float().cpu() for g in values[:len(named)]]
                by_node = {n: g for (n, _), g in zip(active_nodes, values[len(named):])}
                node_report[loss_name] = {n: None if by_node.get(n) is None else float(by_node[n].float().norm())
                                          for n in node_names}
                for n in ('visual_sequence', 'text_sequence'):
                    g = by_node.get(n)
                    if g is not None:
                        g = g.detach().float()
                        if n == 'visual_sequence':
                            detail = dict(cls_norm=float(g[:, :1].norm()), patch_norm=float(g[:, 1:].norm()))
                        else:
                            mask = features['text_content_mask']
                            detail = dict(eot_norm=float(g[torch.arange(len(g), device=g.device), eot].norm()),
                                          content_norm=float(g[mask].norm()),
                                          excluded_norm=float(g[~mask].norm()))
                        node_report[loss_name][n + '_detail'] = detail
                del values, by_node, g
            groups = {}
            for group in ('visual_backbone', 'text_backbone', 'frequency', 'slot_heads', 'relevance_head'):
                indices = [i for i, (name, _) in enumerate(named) if _group(name) == group]
                left, right = ([grads[key][i] for i in indices] for key in ('slot', 'sdm'))
                item = gradient_comparison(left, right)
                item['diversity_norm'] = sum(float(grads['diversity'][i].double().square().sum())
                                             for i in indices if grads['diversity'][i] is not None) ** 0.5
                diversity_vs_slot = gradient_comparison([grads['diversity'][i] for i in indices], left)
                item['diversity_to_slot_norm_ratio'] = diversity_vs_slot['slot_to_sdm_norm_ratio']
                item['diversity_slot_cosine'] = diversity_vs_slot['cosine']
                item['parameter_names'] = [named[i][0] for i in indices]
                groups[group] = item
            per_parameter = {name: gradient_comparison([grads['slot'][i]], [grads['sdm'][i]])
                             for i, (name, _) in enumerate(named)}
            return dict(status='measured', mode='eval, gradients enabled; no augmentation/dropout',
                        losses={k: float(v.detach()) for k, v in losses.items()},
                        weights=dict(slot=model.args.scfp_slot_loss_weight, diversity=model.args.scfp_diversity_weight),
                        parameter_groups=groups, per_parameter=per_parameter, representation_gradients=node_report,
                        batch_statistics=slot_measurement(features, batch['pids'], captured['logits'].squeeze(-1),
                                                          model.logit_scale, batch['caption_ids'], model.args.vocab_size),
                        frozen_parameters=[n for n, p in model.named_parameters() if not p.requires_grad],
                        note='Cosines are for concatenated parameter gradients; None means absent/zero norm. '
                             'Final CLS/EOT node gradients are not a test of earlier shared-parameter connectivity.')
    finally:
        handle.remove()


@torch.no_grad()
def intervention_measurement(model, image_batches, text_embeddings, pids, chunk_size=128):
    frequency = model.base_model.visual.scfp_frequency
    on, off, repeat = [], [], []
    for images in image_batches:
        with frequency_mode(frequency, True):
            on.append(model.encode_image(images).cpu())
        with frequency_mode(frequency, False):
            off.append(model.encode_image(images).cpu())
        with frequency_mode(frequency, True):
            repeat.append(model.encode_image(images).cpu())
    on, off, repeat = (torch.cat(x) for x in (on, off, repeat))
    difference = (on - off).abs()
    norm_on, norm_off = F.normalize(on, dim=-1), F.normalize(off, dim=-1)
    rank_on, rank_off = [], []
    query = F.normalize(text_embeddings.cpu(), dim=-1)
    for start in range(0, len(query), chunk_size):
        q = query[start:start + chunk_size]
        rank_on.append((q @ norm_on.t()).argmax(-1))
        rank_off.append((q @ norm_off.t()).argmax(-1))
    rank_on, rank_off = torch.cat(rank_on), torch.cat(rank_off)
    r1_on = float(pids[rank_on].eq(pids).float().mean()) * 100
    r1_off = float(pids[rank_off].eq(pids).float().mean()) * 100
    return dict(status='measured', global_embedding_modality='image CLS; text EOT encoder is independent of frequency',
                global_embedding_cosine=stats(F.cosine_similarity(on, off, dim=-1)),
                raw_embedding_absolute_difference=stats(difference),
                normalized_embedding_absolute_difference=stats((norm_on - norm_off).abs()),
                raw_embedding_l2_difference=stats((on - off).norm(dim=-1)),
                on_repeat_max_difference=float((on - repeat).abs().max()),
                on_repeat_exact=bool(torch.equal(on, repeat)),
                retrieval=dict(direction='text_to_image', split='fixed_training_subset_seen_during_training',
                               gallery_images=len(on), queries=len(query),
                               rank1_on_percent=r1_on, rank1_off_percent=r1_off,
                               rank1_on_minus_off_percentage_points=r1_on - r1_off,
                               top1_changed_fraction=float(rank_on.ne(rank_off).float().mean()),
                               top1_on=rank_on, top1_off=rank_off,
                               tie_break='First gallery index in the fixed manifest (torch.argmax)',
                               note='Includes each paired image; identity positives; descriptive subset metric only.'),
                interpretation='Inference intervention on the same trained weights. Does not replace a retrained frequency-only ablation.')
