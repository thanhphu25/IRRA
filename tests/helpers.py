from types import SimpleNamespace

import torch
from model.clip_model import CLIP
from utils.options import get_args


def tiny_clip(*args, **kwargs):
    config = dict(embed_dim=512, image_resolution=(32, 16), vision_layers=3,
                  vision_width=64, vision_patch_size=8, stride_size=8,
                  context_length=8, vocab_size=128, transformer_width=64,
                  transformer_heads=1, transformer_layers=1)
    return CLIP(**config), config


def tiny_args(mode='v2'):
    args = get_args([])
    args.scfp_mode = mode
    args.scfp_frequency_layer = 1
    args.vocab_size = 128
    args.cmt_depth = 1
    args.img_size = (32, 16)
    args.text_length = 8
    args.stride_size = 8
    args.precision = 'fp32'
    args.device = 'cpu'
    return args


def synthetic_batch():
    ids = torch.tensor([[126, 3, 4, 127, 0, 0, 0, 0], [126, 4, 5, 127, 0, 0, 0, 0],
                        [126, 6, 7, 127, 0, 0, 0, 0], [126, 8, 9, 127, 0, 0, 0, 0]])
    masked = ids.clone()
    masked[:, 1] = 125
    labels = torch.zeros_like(ids)
    labels[:, 1] = ids[:, 1]
    return dict(images=torch.randn(4, 3, 32, 16), caption_ids=ids, pids=torch.tensor([0, 0, 1, 2]),
                mlm_ids=masked, mlm_labels=labels)
