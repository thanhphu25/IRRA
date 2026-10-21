"""Static plots of measured intermediate responses and head-averaged attention."""
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def display_image(image):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def frequency_plot(path, measurement, images, selected, labels, synthetic=False):
    n = len(selected)
    fig, axes = plt.subplots(n + 1, 8, figsize=(24, 3 * (n + 1)), squeeze=False,
                             constrained_layout=True)
    masks = measurement['masks']
    for k in range(4):
        axes[0, k].imshow(torch.fft.fftshift(masks[k]).numpy(), vmin=0, vmax=1, cmap='gray')
        axes[0, k].set_title(f'Band {k}: {int(masks[k].sum())} FFT bins')
    gains = measurement['band_gains']
    axes[0, 4].plot(gains.numpy().T, alpha=0.15)
    axes[0, 4].plot(gains.mean(0).numpy(), color='black', marker='o')
    axes[0, 4].axhline(1, color='red', linestyle=':')
    axes[0, 4].set_title('Actual gain per image + mean')
    axes[0, 4].set_xticks(range(4))
    axes[0, 5].bar(range(4), measurement['input_energy_fraction'].mean(0).numpy())
    axes[0, 5].set_title('Mean input energy fraction')
    axes[0, 6].bar(range(4), measurement['output_energy'].sum(0).numpy() /
                   measurement['input_energy'].sum(0).clamp_min(1e-30).numpy())
    axes[0, 6].set_title('Output / input band energy')
    axes[0, 7].hist(measurement['residual_ratio'].numpy(), bins=min(20, len(images)))
    axes[0, 7].set_title('Residual / input norm')
    low, high = float(gains.min()), float(gains.max())
    if high - low < 1e-7:
        low, high = low - 1e-5, high + 1e-5
    for row, index in enumerate(selected, 1):
        ax = axes[row]
        ax[0].imshow(display_image(images[index]))
        ax[0].set_title(labels[index], fontsize=8)
        response = torch.fft.fftshift(measurement['response'][index])
        im = ax[1].imshow(response.numpy(), cmap='coolwarm', vmin=low, vmax=high)
        ax[1].set_title('Actual FFT gain (shifted)')
        fig.colorbar(im, ax=ax[1], shrink=0.6)
        maps = measurement['band_rms_maps'][index]
        for k in range(4):
            ax[k + 2].imshow(maps[k].numpy(), vmin=0, vmax=float(maps.max()), cmap='magma')
            ax[k + 2].set_title(f'Band {k} RMS; gain={gains[index, k]:.5f}')
        ax[6].imshow(measurement['residual_rms_map'][index].numpy(), cmap='magma')
        ax[6].set_title(f'Residual RMS; ratio={measurement["residual_ratio"][index]:.5f}')
        ax[7].imshow(measurement['input_rms_map'][index].numpy(), cmap='magma')
        ax[7].set_title('Input feature RMS')
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
    label = 'SYNTHETIC RANDOM MODEL / INPUTS' if synthetic else 'Fixed training subset'
    fig.suptitle(f'{label} | Feature FFT at the actual insertion layer; orthonormal FFT, channel RMS.\n'
                 'Gain scale shared across examples; band RMS scale shared within each example. '
                 'Hard masks partition bins; they are not normalized by band area.', fontsize=12)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def slot_plots(visual_path, text_path, features, images, ids, selected, labels,
               captions, grid, tokenizer=None, synthetic=False):
    n, k = len(selected), features['visual_attention'].shape[1]
    fig, axes = plt.subplots(n, k + 1, figsize=(3 * (k + 1), 3.5 * n),
                             squeeze=False, constrained_layout=True)
    for row, index in enumerate(selected):
        image = display_image(images[index])
        maps = features['visual_attention'][index].reshape(k, *grid)
        scaled = F.interpolate(maps[:, None], size=images.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(labels[index] + '\n' + textwrap.fill(captions[index], 35), fontsize=8)
        for slot in range(k):
            axes[row, slot + 1].imshow(image)
            im = axes[row, slot + 1].imshow(scaled[slot].numpy(), alpha=0.65, cmap='inferno',
                                           vmin=0, vmax=float(scaled.max()))
            axes[row, slot + 1].set_title(f'Slot {slot} | peak patch {int(maps[slot].argmax())}')
        fig.colorbar(im, ax=list(axes[row, 1:]), shrink=0.5)
        for ax in axes[row]:
            ax.set_axis_off()
    label = 'SYNTHETIC RANDOM MODEL / INPUTS' if synthetic else 'Fixed training subset'
    fig.suptitle(f'{label}: four visual slots, head-averaged attention.\n'
                 f'Actual patch grid {grid}; bilinear display only; shared color scale across slots in each row.')
    fig.savefig(visual_path, dpi=125)
    plt.close(fig)
    fig, axes = plt.subplots(n, 1, figsize=(22, 3.4 * n), squeeze=False, constrained_layout=True)
    for row, index in enumerate(selected):
        ax = axes[row, 0]
        valid_indices = features['text_content_mask'][index].nonzero().flatten()
        attention = features['text_attention'][index][:, valid_indices]
        pieces = []
        for position in valid_indices:
            token_id = int(ids[index, position])
            piece = tokenizer.decode([token_id]) if tokenizer else f'id={token_id}'
            pieces.append(f'{int(position)}:{piece.strip()}')
        if attention.shape[-1]:
            im = ax.imshow(attention.numpy(), aspect='auto', cmap='viridis', vmin=0, vmax=float(attention.max()))
            ax.set_xticks(np.arange(len(pieces)), pieces, rotation=55, ha='right', fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.6)
        else:
            ax.text(0.5, 0.5, 'Empty content: sentinel attention is hidden, no content mass')
        ax.set_yticks(range(k), [f'Slot {s}' for s in range(k)])
        ax.set_title(labels[index] + ' | ' + textwrap.fill(captions[index], 150), fontsize=9)
    fig.suptitle(f'{label}: text slot attention over actual unmasked BPE positions.\n'
                 'SOT/EOT/PAD/MASK excluded; shared color scale across all four slots per caption.')
    fig.savefig(text_path, dpi=125)
    plt.close(fig)
